"""Pure-NumPy Vision Transformer (ViT) — port of ``vit.py`` (PyTorch).

Mirrors the module structure of the original repo:
``NewGELUActivation / PatchEmbeddings / Embeddings / AttentionHead /
MultiHeadAttention / FasterMultiHeadAttention / MLP / Block / Encoder /
ViTForClassification`` plus manual backprop, ``CrossEntropyLoss`` and ``AdamW``
so the model can be trained without any deep-learning framework.

Conventions
-----------
* Images: ``x_img`` has shape ``(B, C, H, W)`` (NCHW, float32).
* Sequences: ``x_seq`` has shape ``(B, S, D)`` where ``S`` = num tokens,
  ``D`` = hidden size.
* Every learnable module exposes ``.params`` and ``.grads`` dicts mapping a
  local name to an ``np.ndarray``, plus ``named_parameters(prefix)``,
  ``zero_grad()``, ``state_dict()/load_state_dict()`` helpers.
* ``forward(x, training=True)`` caches what ``backward(dout)`` needs.
  Dropout is identity when ``training=False`` or ``p == 0``.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Small functional helpers (with shape annotations)
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    """Softmax along ``axis``.

    Args:
        x: ndarray ``(..., N)``.

    Returns:
        probs: ndarray, same shape as ``x``; sums to 1 along ``axis``.
    """
    # x: (..., N) -> shift for numerical stability, same shape
    m = x.max(axis=axis, keepdims=True)  # (..., 1)
    e = np.exp(x - m)  # (..., N)
    return e / e.sum(axis=axis, keepdims=True)  # (..., N)


def gelu_forward(x):
    """New-GELU (tanh approximation, BERT/GPT variant).

    Args:
        x: ndarray of any shape.

    Returns:
        y: ndarray, same shape as ``x``.
            ``y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))``
    """
    # x: (...) -> inner/ tanh/inner shapes all (...)
    c = math.sqrt(2.0 / math.pi)  # scalar
    inner = c * (x + 0.044715 * x ** 3)  # (...)
    return 0.5 * x * (1.0 + np.tanh(inner))  # (...)


def gelu_backward(x, dout):
    """Gradient of New-GELU.

    Args:
        x: ndarray ``(...)``, forward input.
        dout: ndarray ``(...)``, upstream gradient.

    Returns:
        dx: ndarray ``(...)``.
    """
    # all shapes (...)
    c = math.sqrt(2.0 / math.pi)
    x3 = x ** 3
    inner = c * (x + 0.044715 * x3)
    tanh_in = np.tanh(inner)  # (...)
    sech2 = 1.0 - tanh_in ** 2  # (...)
    din_dx = c * (1.0 + 3.0 * 0.044715 * x ** 2)  # (...)
    dy_dx = 0.5 * (1.0 + tanh_in) + 0.5 * x * sech2 * din_dx  # (...)
    return dout * dy_dx  # (...)


def layernorm_forward(x, weight, bias, eps=1e-5):
    """LayerNorm over the last axis (no running stats).

    Args:
        x: ``(..., H)``.
        weight: ``(H,)`` scale.
        bias: ``(H,)`` shift.

    Returns:
        y: ``(..., H)`` normalized then scaled/shifted.
        cache: tuple needed by :func:`layernorm_backward`.
    """
    # mean/var: (..., 1); xhat/y: (..., H)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    xhat = (x - mean) / std
    y = xhat * weight + bias
    return y, (x, xhat, mean, var, std, weight, eps)


def layernorm_backward(dout, cache):
    """Backward of :func:`layernorm_forward`.

    Args:
        dout: ``(..., H)`` upstream gradient.
        cache: tuple from forward.

    Returns:
        dx: ``(..., H)``; dweight: ``(H,)``; dbias: ``(H,)``.
    """
    x, xhat, mean, var, std, weight, eps = cache
    # dout/xhat/dx: (..., H); dweight/dbias: (H,)
    H = x.shape[-1]
    dbias = dout.sum(axis=tuple(range(dout.ndim - 1)))
    dweight = (dout * xhat).sum(axis=tuple(range(dout.ndim - 1)))
    dxhat = dout * weight  # (..., H)
    dvar = (dxhat * (x - mean) * -0.5 / (var + eps) ** 1.5).sum(
        axis=-1, keepdims=True)  # (..., 1)
    dmean = (-dxhat / std).sum(axis=-1, keepdims=True)  # (..., 1)
    dx = dxhat / std + dvar * 2.0 * (x - mean) / H + dmean / H  # (..., H)
    return dx, dweight, dbias


def _trunc_normal(rng, shape, mean=0.0, std=0.02, a=-2.0, b=2.0):
    """Approximate truncated normal (clip standard normal to [a, b]).

    Args:
        rng: ``np.random.Generator``.
        shape: output shape tuple.

    Returns:
        ndarray ``shape``, dtype float32.
    """
    z = rng.standard_normal(shape).astype(np.float64)  # shape
    z = np.clip(z, a, b)  # shape
    return (mean + std * z).astype(np.float32)  # shape


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Module:
    """Minimal module base: param bookkeeping + save/load helpers."""

    def params(self):
        """Return dict of learnable parameters (override in subclasses)."""
        return {}

    def grads(self):
        """Return dict of gradients aligned with :meth:`params`."""
        return {}

    def named_parameters(self, prefix=""):
        """Flattened ``{qualified_name: (holder, key)}`` for optimizers.

        Args:
            prefix: string prepended to local param names.
        """
        out = {}
        for k in self.params():
            out[prefix + k] = (self, k)
        return out

    def zero_grad(self):
        """Zero all gradient buffers in-place."""
        for g in self.grads().values():
            g.fill(0)

    def state_dict(self):
        """Return ``{name: ndarray copy}`` of all parameters."""
        return {k: v.copy() for k, v in self.params().items()}

    def load_state_dict(self, *a, **k):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Primitive layers
# ---------------------------------------------------------------------------

class Linear(Module):
    """Affine layer ``y = x @ W + b``.

    Args:
        in_features: input dim.
        out_features: output dim.
        bias: whether to include bias.
        rng: ``np.random.Generator`` for init (or None -> np.random).
        initializer_range: std of N(0, std) weight init.
    """

    def __init__(self, in_features, out_features, bias=True, rng=None,
                 initializer_range=0.02):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        r = rng if rng is not None else np.random
        if hasattr(r, "standard_normal"):
            W = (r.standard_normal((in_features, out_features))
                 .astype(np.float64) * initializer_range)
        else:  # np.random module fallback
            W = r.randn(in_features, out_features) * initializer_range
        self._params = {"weight": W.astype(np.float32)}  # (in, out)
        if bias:
            self._params["bias"] = np.zeros((out_features,), dtype=np.float32)
        self._grads = {k: np.zeros_like(v) for k, v in self._params.items()}
        self._cache = None

    def params(self):
        return self._params

    def grads(self):
        return self._grads

    def forward(self, x, training=True):
        """Forward.

        Args:
            x: ``(..., in)``.

        Returns:
            y: ``(..., out)``.
        """
        # x: (..., in); W: (in, out); y: (..., out)
        self._cache = (x.copy(),)
        y = x @ self._params["weight"]  # (..., out)
        if self.use_bias:
            y = y + self._params["bias"]  # broadcast (out,)
        return y

    def backward(self, dout):
        """Backward.

        Args:
            dout: ``(..., out)`` upstream gradient.

        Returns:
            dx: ``(..., in)``.
        """
        (x,) = self._cache
        # dout: (..., out); flat batch dims for dW accumulation
        xf = x.reshape(-1, self.in_features)  # (M, in)
        df = dout.reshape(-1, self.out_features)  # (M, out)
        self._grads["weight"] += xf.T @ df  # (in, out)
        if self.use_bias:
            self._grads["bias"] += df.sum(axis=0)  # (out,)
        return dout @ self._params["weight"].T  # (..., in)

    def state_dict(self):
        return {k: v.copy() for k, v in self._params.items()}

    def load_state_dict(self, d):
        for k, v in d.items():
            self._params[k][...] = v


class Dropout(Module):
    """Inverted dropout (scales by ``1/(1-p)`` during training).

    Args:
        p: drop probability.
        rng: ``np.random.Generator`` or None (uses global np.random).
    """

    def __init__(self, p=0.0, rng=None):
        self.p = p
        self.rng = rng
        self._mask = None

    def forward(self, x, training=True):
        """Forward.

        Args:
            x: ``(...)`` any shape.

        Returns:
            y: same shape as ``x``.
        """
        if not training or self.p == 0.0:
            self._mask = None
            return x
        keep = 1.0 - self.p
        r = self.rng if self.rng is not None else np.random
        if hasattr(r, "random"):
            mask = (r.random(x.shape) < keep).astype(x.dtype)
        else:
            mask = (r.rand(*x.shape) < keep).astype(x.dtype)
        self._mask = mask  # (...) binary
        return (x * mask) / keep  # (...)

    def backward(self, dout):
        """Backward; ``dx = dout * mask / keep``; shapes like forward."""
        if self._mask is None:
            return dout
        return (dout * self._mask) / (1.0 - self.p)


class LayerNorm(Module):
    """LayerNorm with learnable ``weight``/``bias`` of dim ``normalized_shape``."""

    def __init__(self, normalized_shape, eps=1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self._params = {"weight": np.ones((normalized_shape,), dtype=np.float32),
                        "bias": np.zeros((normalized_shape,), dtype=np.float32)}
        self._grads = {k: np.zeros_like(v) for k, v in self._params.items()}
        self._cache = None

    def params(self):
        return self._params

    def grads(self):
        return self._grads

    def forward(self, x, training=True):
        """Args: x ``(..., H)``. Returns: y ``(..., H)``."""
        y, cache = layernorm_forward(x, self._params["weight"],
                                     self._params["bias"], self.eps)
        self._cache = cache
        return y

    def backward(self, dout):
        """Args: dout ``(..., H)``. Returns: dx ``(..., H)``."""
        dx, dw, db = layernorm_backward(dout, self._cache)
        self._grads["weight"] += dw
        self._grads["bias"] += db
        return dx

    def state_dict(self):
        return {k: v.copy() for k, v in self._params.items()}

    def load_state_dict(self, d):
        for k, v in d.items():
            self._params[k][...] = v


class NewGELUActivation(Module):
    """New-GELU activation (tanh approximation); no parameters."""

    def __init__(self):
        self._cache = None

    def forward(self, x, training=True):
        """Args: x ``(...)``. Returns: y ``(...)`` same shape."""
        self._cache = (x.copy(),)
        return gelu_forward(x)

    def backward(self, dout):
        """Args: dout ``(...)``. Returns: dx ``(...)`` same shape."""
        (x,) = self._cache
        return gelu_backward(x, dout)


# ---------------------------------------------------------------------------
# Patch / token embeddings
# ---------------------------------------------------------------------------

class PatchEmbeddings(Module):
    """Split image into non-overlapping patches and linearly project each.

    Equivalent to ``Conv2d(C, D, kernel=P, stride=P)`` + flatten/transpose.

    Args:
        config: dict with ``image_size, patch_size, num_channels, hidden_size``.
        rng: random generator for weight init.
    """

    def __init__(self, config, rng=None):
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        assert self.image_size % self.patch_size == 0
        self.num_patches = (self.image_size // self.patch_size) ** 2
        r = rng if rng is not None else np.random
        std = config.get("initializer_range", 0.02)
        if hasattr(r, "standard_normal"):
            W = (r.standard_normal((self.hidden_size, self.num_channels,
                                    self.patch_size, self.patch_size))
                 .astype(np.float64) * std).astype(np.float32)
        else:
            W = (r.randn(self.hidden_size, self.num_channels,
                         self.patch_size, self.patch_size) * std).astype(np.float32)
        # W: (D, C, P, P); b: (D,)
        self._params = {"weight": W, "bias": np.zeros((self.hidden_size,), dtype=np.float32)}
        self._grads = {k: np.zeros_like(v) for k, v in self._params.items()}
        self._cache = None

    def params(self):
        return self._params

    def grads(self):
        return self._grads

    def forward(self, x, training=True):
        """Map images to patch tokens.

        Args:
            x: ``(B, C, H, W)`` images.

        Returns:
            tokens: ``(B, N, D)`` with ``N = (H/P)^2`` patches.
        """
        B, C, H, W = x.shape
        P, D = self.patch_size, self.hidden_size
        assert H == self.image_size and W == self.image_size, \
            f"expected {self.image_size}x{self.image_size}, got {H}x{W}"
        n = H // P  # patches per side
        # (B,C,H,W) -> (B,C,n,P,n,P) -> (B,n,n,C,P,P) -> (B,N,C*P*P)
        xp = x.reshape(B, C, n, P, n, P).transpose(0, 2, 4, 1, 3, 5)
        patches = xp.reshape(B, n * n, C * P * P)  # (B, N, C*P*P)
        Wcol = self._params["weight"].reshape(D, C * P * P)  # (D, C*P*P)
        self._cache = (x.shape, patches.copy(), Wcol.copy())
        return patches @ Wcol.T + self._params["bias"]  # (B, N, D)

    def backward(self, dout):
        """Args: dout ``(B, N, D)``. Returns: dx ``(B, C, H, W)``."""
        xshape, patches, Wcol = self._cache
        B, C, H, W = xshape
        P, D = self.patch_size, self.hidden_size
        n = H // P
        # dout: (B,N,D); patches: (B,N,CPP); Wcol: (D,CPP)
        df = dout.reshape(-1, D)  # (B*N, D)
        pf = patches.reshape(-1, C * P * P)  # (B*N, CPP)
        self._grads["weight"] += (df.T @ pf).reshape(D, C, P, P)  # (D,C,P,P)
        self._grads["bias"] += dout.sum(axis=(0, 1))  # (D,)
        dpatches = dout @ Wcol  # (B, N, CPP)
        dx = dpatches.reshape(B, n, n, C, P, P).transpose(0, 3, 1, 4, 2, 5)
        return dx.reshape(B, C, H, W).astype(np.float32)  # (B,C,H,W)

    def state_dict(self):
        return {k: v.copy() for k, v in self._params.items()}

    def load_state_dict(self, d):
        for k, v in d.items():
            self._params[k][...] = v


class Embeddings(Module):
    """Add ``[CLS]`` token + position embeddings to patch tokens.

    Args:
        config: dict with ``hidden_size, hidden_dropout_prob, initializer_range``.
        rng: random generator (truncated-normal init for cls/pos).
    """

    def __init__(self, config, rng=None):
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config, rng=rng)
        D = config["hidden_size"]
        N = self.patch_embeddings.num_patches
        r = rng if rng is not None else np.random.default_rng(0)
        if not hasattr(r, "standard_normal"):
            r = np.random.default_rng(0)
        std = config.get("initializer_range", 0.02)
        # cls: (1,1,D); pos: (1,N+1,D)
        self._params = {
            "cls_token": _trunc_normal(r, (1, 1, D), std=std),
            "position_embeddings": _trunc_normal(r, (1, N + 1, D), std=std),
        }
        self._grads = {k: np.zeros_like(v) for k, v in self._params.items()}
        p = config.get("hidden_dropout_prob", 0.0)
        self.dropout = Dropout(p, rng=rng)
        self._cache = None

    def params(self):
        d = dict(self._params)
        return d

    def grads(self):
        return self._grads

    def named_parameters(self, prefix=""):
        out = {prefix + "cls_token": (self, "cls_token"),
               prefix + "position_embeddings": (self, "position_embeddings")}
        for k, v in self.patch_embeddings.named_parameters(
                prefix + "patch_embeddings.").items():
            out[k] = v
        return out

    def zero_grad(self):
        for g in self._grads.values():
            g.fill(0)
        self.patch_embeddings.zero_grad()

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, value):
        self._params[key][...] = value

    def forward(self, x, training=True):
        """Args: x ``(B, C, H, W)``. Returns: y ``(B, N+1, D)``."""
        # patch_tokens: (B,N,D) -> prepend cls (B,1,D) -> (B,N+1,D)
        patch_tokens = self.patch_embeddings.forward(x, training=training)
        B = patch_tokens.shape[0]
        cls = np.broadcast_to(self._params["cls_token"],
                              (B, 1, self.config["hidden_size"]))  # (B,1,D)
        seq = np.concatenate([cls, patch_tokens], axis=1)  # (B,N+1,D)
        seq = seq + self._params["position_embeddings"]  # broadcast (1,N+1,D)
        self._cache = (B, patch_tokens.shape)
        return self.dropout.forward(seq, training=training)  # (B,N+1,D)

    def backward(self, dout):
        """Args: dout ``(B, N+1, D)``. Returns: dx ``(B, C, H, W)``."""
        B, _ = self._cache
        dseq = self.dropout.backward(dout)  # (B,N+1,D)
        # pos grad broadcasts over batch: sum axis 0 keep (1,N+1,D)
        self._grads["position_embeddings"] += dseq.sum(axis=0, keepdims=True)
        # cls grad: first token summed over batch -> (1,1,D)
        self._grads["cls_token"] += dseq[:, 0:1, :].sum(axis=0, keepdims=True)
        dpatches = dseq[:, 1:, :]  # (B,N,D)
        return self.patch_embeddings.backward(dpatches)  # (B,C,H,W)

    def state_dict(self):
        d = {k: v.copy() for k, v in self._params.items()}
        d["patch_embeddings.weight"] = self.patch_embeddings.params()["weight"].copy()
        d["patch_embeddings.bias"] = self.patch_embeddings.params()["bias"].copy()
        return d

    def load_state_dict(self, d):
        for k in self._params:
            if k in d:
                self._params[k][...] = d[k]
        pd = {}
        if "patch_embeddings.weight" in d:
            pd["weight"] = d["patch_embeddings.weight"]
        if "patch_embeddings.bias" in d:
            pd["bias"] = d["patch_embeddings.bias"]
        if pd:
            self.patch_embeddings.load_state_dict(pd)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class AttentionHead(Module):
    """Single self-attention head: ``softmax(QK^T/sqrt(d))V``.

    Args:
        hidden_size: input dim ``D``.
        attention_head_size: per-head dim ``d``.
        dropout: attention-probs dropout prob.
        bias: use bias in Q/K/V linears.
        rng: random generator.
    """

    def __init__(self, hidden_size, attention_head_size, dropout, bias=True, rng=None):
        self.attention_head_size = attention_head_size
        self.query = Linear(hidden_size, attention_head_size, bias=bias, rng=rng)
        self.key = Linear(hidden_size, attention_head_size, bias=bias, rng=rng)
        self.value = Linear(hidden_size, attention_head_size, bias=bias, rng=rng)
        self.dropout = Dropout(dropout, rng=rng)
        self._cache = None

    def named_parameters(self, prefix=""):
        out = {}
        for k, v in self.query.named_parameters(prefix + "query.").items():
            out[k] = v
        for k, v in self.key.named_parameters(prefix + "key.").items():
            out[k] = v
        for k, v in self.value.named_parameters(prefix + "value.").items():
            out[k] = v
        return out

    def zero_grad(self):
        self.query.zero_grad()
        self.key.zero_grad()
        self.value.zero_grad()

    def forward(self, x, training=True):
        """Args: x ``(B, S, D)``. Returns: ``(out (B,S,d), probs (B,S,S))``."""
        # q/k/v: (B,S,d); scores/probs: (B,S,S); out: (B,S,d)
        q = self.query.forward(x, training=training)
        k = self.key.forward(x, training=training)
        v = self.value.forward(x, training=training)
        scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.attention_head_size)
        probs = softmax(scores, axis=-1)
        probs_d = self.dropout.forward(probs, training=training)
        out = probs_d @ v
        self._cache = (q, k, v, probs_d.copy(), probs.copy())
        return out, probs_d

    def backward(self, dout, dprobs_unused=None):
        """Args: dout ``(B,S,d)``. Returns: dx ``(B,S,D)`` (summed over Q/K/V paths)."""
        q, k, v, probs_d, probs = self._cache
        # out = P @ V with P=(B,S,S), V=(B,S,d)
        dP = dout @ v.transpose(0, 2, 1)  # (B,S,S)
        dV = probs_d.transpose(0, 2, 1) @ dout  # (B,S,d)
        dP = self.dropout.backward(dP)  # (B,S,S)
        # softmax backward: dS = P*(dP - sum(dP*P))
        dS = probs * (dP - (dP * probs).sum(axis=-1, keepdims=True))  # (B,S,S)
        scale = 1.0 / math.sqrt(self.attention_head_size)
        dQ = (dS @ k) * scale  # (B,S,d)
        dK = (dS.transpose(0, 2, 1) @ q) * scale  # (B,S,d)
        dx = self.query.backward(dQ) + self.key.backward(dK) + self.value.backward(dV)
        return dx  # (B,S,D)

    def state_dict(self):
        return {"query." + k: v.copy() for k, v in self.query.state_dict().items()} | \
               {"key." + k: v.copy() for k, v in self.key.state_dict().items()} | \
               {"value." + k: v.copy() for k, v in self.value.state_dict().items()}

    def load_state_dict(self, d):
        self.query.load_state_dict({k[6:]: v for k, v in d.items() if k.startswith("query.")})
        self.key.load_state_dict({k[4:]: v for k, v in d.items() if k.startswith("key.")})
        self.value.load_state_dict({k[6:]: v for k, v in d.items() if k.startswith("value.")})


class MultiHeadAttention(Module):
    """Multi-head attention with one :class:`AttentionHead` per head + output proj.

    Args:
        config: dict with ``hidden_size, num_attention_heads, qkv_bias,
            attention_probs_dropout_prob, hidden_dropout_prob``.
        rng: random generator.
    """

    def __init__(self, config, rng=None):
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.heads = [AttentionHead(self.hidden_size, self.attention_head_size,
                                    config.get("attention_probs_dropout_prob", 0.0),
                                    config.get("qkv_bias", True), rng=rng)
                      for _ in range(self.num_attention_heads)]
        self.output_projection = Linear(self.all_head_size, self.hidden_size,
                                        rng=rng)
        self.output_dropout = Dropout(config.get("hidden_dropout_prob", 0.0), rng=rng)
        self._cache = None

    def named_parameters(self, prefix=""):
        out = {}
        for i, h in enumerate(self.heads):
            for k, v in h.named_parameters(prefix + f"heads.{i}.").items():
                out[k] = v
        for k, v in self.output_projection.named_parameters(
                prefix + "output_projection.").items():
            out[k] = v
        return out

    def zero_grad(self):
        for h in self.heads:
            h.zero_grad()
        self.output_projection.zero_grad()

    def forward(self, x, output_attentions=False, training=True):
        """Args: x ``(B,S,D)``. Returns: ``(out (B,S,D), probs or None)``.

        ``probs`` stacks per-head maps as ``(B, H, S, S)`` when requested.
        """
        outs, probs = [], []
        for h in self.heads:
            o, p = h.forward(x, training=training)  # o:(B,S,d) p:(B,S,S)
            outs.append(o)
            probs.append(p)
        concat = np.concatenate(outs, axis=-1)  # (B,S,H*d)=(B,S,all_head)
        proj = self.output_projection.forward(concat, training=training)
        out = self.output_dropout.forward(proj, training=training)  # (B,S,D)
        self._cache = (concat.shape,)
        if not output_attentions:
            return out, None
        return out, np.stack(probs, axis=1)  # (B,H,S,S)

    def backward(self, dout):
        """Args: dout ``(B,S,D)``. Returns: dx ``(B,S,D)``."""
        d = self.output_dropout.backward(dout)  # (B,S,D)
        dconcat = self.output_projection.backward(d)  # (B,S,all_head)
        parts = np.split(dconcat, self.num_attention_heads, axis=-1)  # H x (B,S,d)
        dx = 0
        for h, dp in zip(self.heads, parts):
            dx = dx + h.backward(dp)  # (B,S,D)
        return dx

    def state_dict(self):
        d = {}
        for i, h in enumerate(self.heads):
            for k, v in h.state_dict().items():
                d[f"heads.{i}.{k}"] = v
        for k, v in self.output_projection.state_dict().items():
            d[f"output_projection.{k}"] = v
        return d

    def load_state_dict(self, d):
        for i, h in enumerate(self.heads):
            h.load_state_dict({k[len(f"heads.{i}."):]: v for k, v in d.items()
                               if k.startswith(f"heads.{i}.")})
        self.output_projection.load_state_dict(
            {k[len("output_projection."):]: v for k, v in d.items()
             if k.startswith("output_projection.")})


class FasterMultiHeadAttention(Module):
    """Fused multi-head attention (single QKV projection), math-identical output.

    Args:
        config: same keys as :class:`MultiHeadAttention`.
        rng: random generator.
    """

    def __init__(self, config, rng=None):
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_projection = Linear(self.hidden_size, self.all_head_size * 3,
                                     bias=config.get("qkv_bias", True), rng=rng)
        self.attn_dropout = Dropout(config.get("attention_probs_dropout_prob", 0.0), rng=rng)
        self.output_projection = Linear(self.all_head_size, self.hidden_size, rng=rng)
        self.output_dropout = Dropout(config.get("hidden_dropout_prob", 0.0), rng=rng)
        self._cache = None

    def named_parameters(self, prefix=""):
        out = {}
        for k, v in self.qkv_projection.named_parameters(prefix + "qkv_projection.").items():
            out[k] = v
        for k, v in self.output_projection.named_parameters(
                prefix + "output_projection.").items():
            out[k] = v
        return out

    def zero_grad(self):
        self.qkv_projection.zero_grad()
        self.output_projection.zero_grad()

    def _split_heads(self, t):
        """``(B,S,H*d)`` -> ``(B,H,S,d)``."""
        B, S, _ = t.shape
        return t.reshape(B, S, self.num_attention_heads,
                         self.attention_head_size).transpose(0, 2, 1, 3)

    def _merge_heads(self, t):
        """``(B,H,S,d)`` -> ``(B,S,H*d)``."""
        B, H, S, d = t.shape
        return t.transpose(0, 2, 1, 3).reshape(B, S, H * d)

    def forward(self, x, output_attentions=False, training=True):
        """Args: x ``(B,S,D)``. Returns: ``(out (B,S,D), probs (B,H,S,S)|None)``."""
        # qkv: (B,S,3*all); q/k/v: (B,H,S,d); scores/probs: (B,H,S,S)
        qkv = self.qkv_projection.forward(x, training=training)
        A = self.all_head_size
        q, k, v = qkv[..., :A], qkv[..., A:2 * A], qkv[..., 2 * A:]
        qh, kh, vh = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        scores = (qh @ kh.transpose(0, 1, 3, 2)) / math.sqrt(self.attention_head_size)
        probs = softmax(scores, axis=-1)  # (B,H,S,S)
        probs_d = self.attn_dropout.forward(probs, training=training)
        oh = probs_d @ vh  # (B,H,S,d)
        merged = self._merge_heads(oh)  # (B,S,all)
        proj = self.output_projection.forward(merged, training=training)
        out = self.output_dropout.forward(proj, training=training)  # (B,S,D)
        self._cache = (qh.copy(), kh.copy(), vh.copy(), probs.copy(),
                       probs_d.copy(), oh.copy(), merged.shape, qkv.shape)
        if not output_attentions:
            return out, None
        return out, probs_d

    def backward(self, dout):
        """Args: dout ``(B,S,D)``. Returns: dx ``(B,S,D)``."""
        qh, kh, vh, probs, probs_d, oh, merged_shape, qkv_shape = self._cache
        d = self.output_dropout.backward(dout)  # (B,S,D)
        dmerged = self.output_projection.backward(d)  # (B,S,all)
        B, H, S, dh = oh.shape
        doh = dmerged.reshape(B, S, H, dh).transpose(0, 2, 1, 3)  # (B,H,S,d)
        dP = doh @ vh.transpose(0, 1, 3, 2)  # (B,H,S,S)
        dV = probs_d.transpose(0, 1, 3, 2) @ doh  # (B,H,S,d)
        dP = self.attn_dropout.backward(dP)
        dS = probs * (dP - (dP * probs).sum(axis=-1, keepdims=True))  # (B,H,S,S)
        scale = 1.0 / math.sqrt(self.attention_head_size)
        dQ = (dS @ kh) * scale  # (B,H,S,d)
        dK = (dS.transpose(0, 1, 3, 2) @ qh) * scale  # (B,H,S,d)
        # merge head grads -> (B,S,3*all)
        def merge(t):
            return t.transpose(0, 2, 1, 3).reshape(B, S, H * dh)
        dqkv = np.concatenate([merge(dQ), merge(dK), merge(dV)], axis=-1)
        return self.qkv_projection.backward(dqkv)  # (B,S,D)

    def state_dict(self):
        return ({"qkv_projection." + k: v.copy()
                 for k, v in self.qkv_projection.state_dict().items()} |
                {"output_projection." + k: v.copy()
                 for k, v in self.output_projection.state_dict().items()})

    def load_state_dict(self, d):
        self.qkv_projection.load_state_dict(
            {k[15:]: v for k, v in d.items() if k.startswith("qkv_projection.")})
        self.output_projection.load_state_dict(
            {k[18:]: v for k, v in d.items() if k.startswith("output_projection.")})


# ---------------------------------------------------------------------------
# MLP / Block / Encoder / ViT
# ---------------------------------------------------------------------------

class MLP(Module):
    """Two-layer feed-forward net: ``D -> I -> GELU -> D`` with dropout.

    Args:
        config: dict with ``hidden_size, intermediate_size, hidden_dropout_prob``.
        rng: random generator.
    """

    def __init__(self, config, rng=None):
        self.dense_1 = Linear(config["hidden_size"], config["intermediate_size"], rng=rng)
        self.activation = NewGELUActivation()
        self.dense_2 = Linear(config["intermediate_size"], config["hidden_size"], rng=rng)
        self.dropout = Dropout(config.get("hidden_dropout_prob", 0.0), rng=rng)

    def named_parameters(self, prefix=""):
        out = {}
        for k, v in self.dense_1.named_parameters(prefix + "dense_1.").items():
            out[k] = v
        for k, v in self.dense_2.named_parameters(prefix + "dense_2.").items():
            out[k] = v
        return out

    def zero_grad(self):
        self.dense_1.zero_grad()
        self.dense_2.zero_grad()

    def forward(self, x, training=True):
        """Args: x ``(B,S,D)``. Returns: y ``(B,S,D)``."""
        # (B,S,D) -> (B,S,I) -> GELU -> (B,S,D) -> dropout
        h = self.dense_1.forward(x, training=training)
        h = self.activation.forward(h, training=training)
        h = self.dense_2.forward(h, training=training)
        return self.dropout.forward(h, training=training)

    def backward(self, dout):
        """Args: dout ``(B,S,D)``. Returns: dx ``(B,S,D)``."""
        d = self.dropout.backward(dout)
        d = self.dense_2.backward(d)
        d = self.activation.backward(d)
        return self.dense_1.backward(d)

    def state_dict(self):
        return ({"dense_1." + k: v.copy() for k, v in self.dense_1.state_dict().items()} |
                {"dense_2." + k: v.copy() for k, v in self.dense_2.state_dict().items()})

    def load_state_dict(self, d):
        self.dense_1.load_state_dict({k[8:]: v for k, v in d.items() if k.startswith("dense_1.")})
        self.dense_2.load_state_dict({k[8:]: v for k, v in d.items() if k.startswith("dense_2.")})


class Block(Module):
    """Pre-LN transformer block: ``x + Attn(LN1(x))`` then ``x + MLP(LN2(x))``."""

    def __init__(self, config, rng=None):
        self.use_faster_attention = config.get("use_faster_attention", False)
        Attn = FasterMultiHeadAttention if self.use_faster_attention else MultiHeadAttention
        self.attention = Attn(config, rng=rng)
        self.layernorm_1 = LayerNorm(config["hidden_size"])
        self.mlp = MLP(config, rng=rng)
        self.layernorm_2 = LayerNorm(config["hidden_size"])
        self._cache = None

    def named_parameters(self, prefix=""):
        out = {}
        for k, v in self.attention.named_parameters(prefix + "attention.").items():
            out[k] = v
        for k, v in self.layernorm_1.named_parameters(prefix + "layernorm_1.").items():
            out[k] = v
        for k, v in self.mlp.named_parameters(prefix + "mlp.").items():
            out[k] = v
        for k, v in self.layernorm_2.named_parameters(prefix + "layernorm_2.").items():
            out[k] = v
        return out

    def zero_grad(self):
        self.attention.zero_grad()
        self.layernorm_1.zero_grad()
        self.mlp.zero_grad()
        self.layernorm_2.zero_grad()

    def forward(self, x, output_attentions=False, training=True):
        """Args: x ``(B,S,D)``. Returns: ``(y (B,S,D), probs|None)``."""
        h = self.layernorm_1.forward(x, training=training)  # (B,S,D)
        a, probs = self.attention.forward(h, output_attentions=output_attentions,
                                          training=training)  # (B,S,D)
        x = x + a  # residual (B,S,D)
        h2 = self.layernorm_2.forward(x, training=training)  # (B,S,D)
        m = self.mlp.forward(h2, training=training)  # (B,S,D)
        return x + m, probs  # (B,S,D)

    def backward(self, dout):
        """Args: dout ``(B,S,D)``. Returns: dx ``(B,S,D)``."""
        # y = x_res + MLP(LN2(x_res)); x_res = x + Attn(LN1(x))
        dm = self.mlp.backward(dout)  # (B,S,D)
        dx_res = self.layernorm_2.backward(dm) + dout  # (B,S,D)
        da = self.attention.backward(dx_res)  # (B,S,D)
        return self.layernorm_1.backward(da) + dx_res  # (B,S,D)

    def state_dict(self):
        d = {}
        for k, v in self.attention.state_dict().items():
            d[f"attention.{k}"] = v
        for k, v in self.layernorm_1.state_dict().items():
            d[f"layernorm_1.{k}"] = v
        for k, v in self.mlp.state_dict().items():
            d[f"mlp.{k}"] = v
        for k, v in self.layernorm_2.state_dict().items():
            d[f"layernorm_2.{k}"] = v
        return d

    def load_state_dict(self, d):
        self.attention.load_state_dict({k[10:]: v for k, v in d.items()
                                        if k.startswith("attention.")})
        self.layernorm_1.load_state_dict({k[12:]: v for k, v in d.items()
                                          if k.startswith("layernorm_1.")})
        self.mlp.load_state_dict({k[4:]: v for k, v in d.items() if k.startswith("mlp.")})
        self.layernorm_2.load_state_dict({k[12:]: v for k, v in d.items()
                                          if k.startswith("layernorm_2.")})


class Encoder(Module):
    """Stack of ``num_hidden_layers`` transformer blocks."""

    def __init__(self, config, rng=None):
        self.blocks = [Block(config, rng=rng)
                       for _ in range(config["num_hidden_layers"])]

    def named_parameters(self, prefix=""):
        out = {}
        for i, b in enumerate(self.blocks):
            for k, v in b.named_parameters(prefix + f"blocks.{i}.").items():
                out[k] = v
        return out

    def zero_grad(self):
        for b in self.blocks:
            b.zero_grad()

    def forward(self, x, output_attentions=False, training=True):
        """Args: x ``(B,S,D)``. Returns: ``(y (B,S,D), [probs per layer]|None)``."""
        all_attn = [] if output_attentions else None
        for b in self.blocks:
            x, p = b.forward(x, output_attentions=output_attentions, training=training)
            if output_attentions:
                all_attn.append(p)
        return x, all_attn

    def backward(self, dout):
        """Args: dout ``(B,S,D)``. Returns: dx ``(B,S,D)``."""
        for b in reversed(self.blocks):
            dout = b.backward(dout)
        return dout

    def state_dict(self):
        d = {}
        for i, b in enumerate(self.blocks):
            for k, v in b.state_dict().items():
                d[f"blocks.{i}.{k}"] = v
        return d

    def load_state_dict(self, d):
        for i, b in enumerate(self.blocks):
            b.load_state_dict({k[len(f"blocks.{i}."):]: v for k, v in d.items()
                               if k.startswith(f"blocks.{i}.")})


class ViTForClassification(Module):
    """ViT classifier: embeddings -> encoder -> linear head on ``[CLS]``.

    Args:
        config: full model config dict (see ``train_numpy.py``).
        rng: ``np.random.Generator`` or seed int or None.
    """

    def __init__(self, config, rng=None):
        self.config = dict(config)
        if isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(int(rng))
        if rng is None:
            rng = np.random.default_rng(0)
        self._rng = rng
        self.embedding = Embeddings(config, rng=rng)
        self.encoder = Encoder(config, rng=rng)
        self.classifier = Linear(config["hidden_size"], config["num_classes"], rng=rng)
        self._init_weights()

    # -- bookkeeping -----------------------------------------------------
    def named_parameters(self, prefix=""):
        out = {}
        for k, v in self.embedding.named_parameters(prefix + "embedding.").items():
            out[k] = v
        for k, v in self.encoder.named_parameters(prefix + "encoder.").items():
            out[k] = v
        for k, v in self.classifier.named_parameters(prefix + "classifier.").items():
            out[k] = v
        return out

    def zero_grad(self):
        self.embedding.zero_grad()
        self.encoder.zero_grad()
        self.classifier.zero_grad()

    def _init_weights(self):
        """Re-init Linear/Conv weights ``N(0, init_range)`` (biases stay 0)."""
        std = self.config.get("initializer_range", 0.02)
        for _, (holder, key) in self.named_parameters().items():
            if isinstance(holder, Linear) and key == "weight":
                holder.params()["weight"][...] = (
                    self._rng.standard_normal(
                        holder.params()["weight"].shape).astype(np.float64) * std
                ).astype(np.float32)

    # -- forward/backward ------------------------------------------------
    def forward(self, x, output_attentions=False, training=True):
        """Args: x ``(B,C,H,W)``. Returns: ``(logits (B,K), attentions|None)``."""
        # emb: (B,N+1,D) -> enc: (B,N+1,D) -> cls (B,D) -> logits (B,K)
        emb = self.embedding.forward(x, training=training)
        enc, attns = self.encoder.forward(emb, output_attentions=output_attentions,
                                          training=training)
        cls = enc[:, 0, :]  # (B,D)
        self._cache = (enc.shape,)
        logits = self.classifier.forward(cls, training=training)  # (B,K)
        if not output_attentions:
            return logits, None
        return logits, attns

    def backward(self, dlogits):
        """Args: dlogits ``(B,K)``. Returns: dx ``(B,C,H,W)`` (usually unused)."""
        (enc_shape,) = self._cache
        dcls = self.classifier.backward(dlogits)  # (B,D)
        denc = np.zeros(enc_shape, dtype=np.float32)  # (B,N+1,D)
        denc[:, 0, :] = dcls
        demb = self.encoder.backward(denc)  # (B,N+1,D)
        return self.embedding.backward(demb)  # (B,C,H,W)

    def state_dict(self):
        d = {}
        for k, v in self.embedding.state_dict().items():
            d[f"embedding.{k}"] = v
        for k, v in self.encoder.state_dict().items():
            d[f"encoder.{k}"] = v
        for k, v in self.classifier.state_dict().items():
            d[f"classifier.{k}"] = v
        return d

    def load_state_dict(self, d):
        self.embedding.load_state_dict({k[10:]: v for k, v in d.items()
                                        if k.startswith("embedding.")})
        self.encoder.load_state_dict({k[8:]: v for k, v in d.items()
                                      if k.startswith("encoder.")})
        self.classifier.load_state_dict({k[11:]: v for k, v in d.items()
                                         if k.startswith("classifier.")})

    def save(self, path):
        """Save parameters to ``.npz``."""
        np.savez(path, **{k: v for k, v in self.state_dict().items()})

    @classmethod
    def load(cls, path, config):
        """Load ``.npz`` checkpoint into a fresh model."""
        model = cls(config)
        z = np.load(path, allow_pickle=False)
        model.load_state_dict({k: z[k] for k in z.files})
        return model


# ---------------------------------------------------------------------------
# Loss + optimizer
# ---------------------------------------------------------------------------

class CrossEntropyLoss:
    """Mean softmax cross-entropy over a batch (no parameters)."""

    def __init__(self):
        self._cache = None

    def forward(self, logits, labels):
        """Args: logits ``(B,K)`` float; labels ``(B,)`` int.

        Returns:
            loss: scalar float (mean NLL).
        """
        # shifted: (B,K); probs: (B,K); nll: (B,)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        nll = -np.log(probs[np.arange(len(labels)), labels] + 1e-12)
        self._cache = (probs.copy(), labels.copy())
        return float(nll.mean())

    def backward(self):
        """Returns: dlogits ``(B,K)`` = ``(probs - onehot)/B``."""
        probs, labels = self._cache
        d = probs  # (B,K)
        d[np.arange(len(labels)), labels] -= 1.0
        return d / len(labels)

    def __call__(self, logits, labels):
        return self.forward(logits, labels)


class AdamW:
    """Adam with decoupled weight decay over a ViT model's parameters.

    Args:
        model: :class:`ViTForClassification` (params updated in-place).
        lr: learning rate.
        weight_decay: decoupled L2 coefficient.
        betas: ``(beta1, beta2)``.
        eps: numerical stability term.
    """

    def __init__(self, model, lr=1e-2, weight_decay=1e-2, betas=(0.9, 0.999),
                 eps=1e-8):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self._state = {}  # name -> (m, v) ndarrays, same shape as param

    def zero_grad(self):
        """Zero all model gradients."""
        self.model.zero_grad()

    def step(self):
        """One AdamW update in-place; shapes preserved per parameter."""
        self.t += 1
        for name, (holder, key) in self.model.named_parameters().items():
            p = holder.params()[key]  # (...) any shape
            g = holder.grads()[key]  # same shape
            if name not in self._state:
                self._state[name] = (np.zeros_like(p), np.zeros_like(p))
            m, v = self._state[name]
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * g * g
            mhat = m / (1 - self.beta1 ** self.t)
            vhat = v / (1 - self.beta2 ** self.t)
            p *= (1 - self.lr * self.weight_decay)  # decoupled decay
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
