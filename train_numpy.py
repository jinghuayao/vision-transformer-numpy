"""NumPy ViT trainer — port of ``train.py`` (no torch).

Uses :class:`vit_numpy.ViTForClassification` with manual backprop,
:class:`vit_numpy.CrossEntropyLoss` and :class:`vit_numpy.AdamW`, plus the
NumPy CIFAR-10 loader. Run e.g.::

    python train_numpy.py --exp-name vit-numpy-test --epochs 2 --batch-size 32
"""

import argparse
import os
import numpy as np

from vit_numpy import ViTForClassification, CrossEntropyLoss, AdamW
from data_numpy import prepare_data
from utils_numpy import save_experiment, save_checkpoint

config = {
    "patch_size": 4,          # 32x32 image -> 8x8 = 64 patches
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config["intermediate_size"] == 4 * config["hidden_size"]
assert config["image_size"] % config["patch_size"] == 0


class Trainer:
    """Simple supervised trainer (NumPy port of ``train.Trainer``).

    Args:
        model: :class:`ViTForClassification`.
        optimizer: :class:`AdamW` bound to ``model``.
        loss_fn: :class:`CrossEntropyLoss`.
        exp_name: experiment folder name for checkpoints.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """Train for ``epochs``; returns ``(train_losses, test_losses, accuracies)``."""
        train_losses, test_losses, accuracies = [], [], []
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, "
                  f"Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}", flush=True)
            if (save_model_every_n_epochs > 0 and (i + 1) % save_model_every_n_epochs == 0
                    and i + 1 != epochs):
                print("\tSave checkpoint at epoch", i + 1, flush=True)
                save_checkpoint(self.exp_name, self.model, i + 1)
        save_experiment(self.exp_name, self.model.config, self.model,
                        train_losses, test_losses, accuracies)
        return train_losses, test_losses, accuracies

    def train_epoch(self, trainloader):
        """One SGD epoch over ``trainloader``; returns mean loss per sample.

        Per batch ``(B,3,32,32)``: forward (training=True) -> loss ->
        ``model.zero_grad()`` -> ``loss.backward`` -> ``model.backward`` ->
        ``optimizer.step()``.
        """
        total_loss, total_n = 0.0, 0
        for images, labels in trainloader:
            # images: (B,3,32,32); labels: (B,)
            logits, _ = self.model.forward(images, training=True)  # (B,K)
            loss = self.loss_fn.forward(logits, labels)  # scalar
            self.model.zero_grad()
            dlogits = self.loss_fn.backward()  # (B,K)
            self.model.backward(dlogits)
            self.optimizer.step()
            total_loss += loss * len(images)
            total_n += len(images)
        return total_loss / max(total_n, 1)

    def evaluate(self, testloader):
        """Mean accuracy and loss over ``testloader`` (forward with training=False)."""
        total_loss, correct, total_n = 0.0, 0, 0
        for images, labels in testloader:
            logits, _ = self.model.forward(images, training=False)  # (B,K)
            loss = self.loss_fn.forward(logits, labels)
            preds = logits.argmax(axis=1)  # (B,)
            correct += int((preds == labels).sum())
            total_loss += loss * len(images)
            total_n += len(images)
        return correct / max(total_n, 1), total_loss / max(total_n, 1)


def parse_args():
    """CLI args mirroring ``train.py`` (minus torch device handling)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--save-model-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--data-root", type=str,
                        default=os.environ.get("VIT_DATA_ROOT", "./data"))
    return parser.parse_args()


def main():
    """Build data/model/optimizer and run training (port of ``train.main``)."""
    args = parse_args()
    np.random.seed(args.seed)
    trainloader, testloader, _ = prepare_data(
        batch_size=args.batch_size, train_sample_size=args.train_samples,
        test_sample_size=args.test_samples, data_root=args.data_root, seed=args.seed)
    model = ViTForClassification(config, rng=np.random.default_rng(args.seed))
    optimizer = AdamW(model, lr=args.lr, weight_decay=1e-2)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name)
    trainer.train(trainloader, testloader, args.epochs,
                  save_model_every_n_epochs=args.save_model_every)


if __name__ == "__main__":
    main()
