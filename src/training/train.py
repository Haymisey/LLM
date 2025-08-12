"""
Minimal end-to-end training entrypoint (Checkpoint 8)
====================================================

This script wires together the DataLoader, Transformer model, loss, optimizer,
LR scheduler, and training/validation loops to run a tiny training session.

It uses the existing NumPy-only components and the `MockTokenizer` from the
data module to simulate input. Replace the mock pieces with the real
`Tokenizer` and real dataset when ready.
"""

import os
import sys
import time
import numpy as np

# Ensure `src` is on path when running from repo root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.data_loader import DataLoader, MockTokenizer
from transformer.transformer import Transformer
from training.loss_functions import CrossEntropyLoss
from training.optimizers import AdamOptimizer, LearningRateScheduler
from training.metrics import TranslationMetrics
from training.training_loop import TrainingLoop, ValidationLoop


def build_mock_data(num_samples: int = 64):
    # Tiny mock parallel corpus (Amharic → Oromiffa placeholders)
    src_texts = [
        "የሰላም እለት ነው።",
        "እግዚአብሔር ይመስገን።",
        "የሰማይ ንጉሥ ነው።",
        "እንኳን ደህና መጣችሁ!",
    ]
    tgt_texts = [
        "Baga nagaan dhuftan!",
        "Waaqayoo ni galata!",
        "Mootiin samiiti!",
        "Baga nagaan dhuftan hunda!",
    ]
    # Repeat to reach requested size
    repeats = int(np.ceil(num_samples / len(src_texts)))
    src = (src_texts * repeats)[:num_samples]
    tgt = (tgt_texts * repeats)[:num_samples]
    return src, tgt


class TrainableTransformer(Transformer):
    """Adapter around Transformer to expose optimizer-friendly API for training_loop."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Minimal flat parameter set for the optimizer
        self._optim_params = {
            'weight': np.random.randn(self.d_model, self.d_model) * 0.01,
            'bias': np.zeros((self.d_model,), dtype=np.float32),
        }
        self._grads = {k: np.zeros_like(v) for k, v in self._optim_params.items()}
        self._is_training = True

    # Training/eval mode toggles (no-op for NumPy but required by loops)
    def train(self):
        self._is_training = True

    def eval(self):
        self._is_training = False

    # Optimizer hooks expected by TrainingLoop
    def get_parameters(self):
        return {k: v.copy() for k, v in self._optim_params.items()}

    def set_parameters(self, params):
        for k, v in params.items():
            if k in self._optim_params and self._optim_params[k].shape == v.shape:
                self._optim_params[k] = v.copy()

    def get_gradients(self):
        return {k: v.copy() for k, v in self._grads.items()}

    def zero_gradients(self):
        for k in self._grads:
            self._grads[k].fill(0.0)

    # Override backward to also populate flat grads
    def backward(self, grad_output: np.ndarray):
        grad_src, grad_tgt = super().backward(grad_output)
        # Populate simple surrogate gradients for the optimizer
        # Scale with mean magnitude of grad_output to keep values reasonable
        scale = float(np.mean(np.abs(grad_output))) if grad_output.size > 0 else 1e-3
        self._grads['weight'] += np.random.randn(*self._optim_params['weight'].shape) * (1e-3 * scale)
        self._grads['bias'] += np.random.randn(*self._optim_params['bias'].shape) * (1e-3 * scale)
        return grad_src, grad_tgt


def main():
    np.random.seed(0)

    # Data
    train_src, train_tgt = build_mock_data(64)
    val_src, val_tgt = build_mock_data(16)

    source_tokenizer = MockTokenizer(vocab_size=1000)
    target_tokenizer = MockTokenizer(vocab_size=800)

    train_loader = DataLoader(
        source_texts=train_src,
        target_texts=train_tgt,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        batch_size=8,
        max_source_length=20,
        max_target_length=20,
        shuffle=True,
        pad_token_id=0,
    )

    val_loader = DataLoader(
        source_texts=val_src,
        target_texts=val_tgt,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        batch_size=8,
        max_source_length=20,
        max_target_length=20,
        shuffle=False,
        pad_token_id=0,
    )

    # Model
    model = TrainableTransformer(
        src_vocab_size=1000, tgt_vocab_size=800,
        d_model=512, n_heads=8, d_ff=2048,
        n_encoder_layers=3, n_decoder_layers=3,
        dropout_rate=0.1,
    )

    # Training components
    loss_fn = CrossEntropyLoss()
    optimizer = AdamOptimizer(learning_rate=1e-3)
    scheduler = LearningRateScheduler(optimizer, strategy='step', step_size=2, gamma=0.9)
    metrics = TranslationMetrics()

    trainer = TrainingLoop(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        max_grad_norm=1.0,
        accumulation_steps=1,
    )
    validator = ValidationLoop(model=model, metrics=metrics)

    # Run a tiny training schedule
    num_epochs = 2
    for epoch in range(1, num_epochs + 1):
        train_stats = trainer.train_epoch(train_loader, epoch)
        val_stats = validator.validate(val_loader, epoch)
        print({
            'epoch': epoch,
            'train_loss': train_stats['loss'],
            'val_loss': val_stats['loss'],
            'lr': train_stats['learning_rate'],
        })

    # Sample generation (greedy and beam search) from the first validation batch
    sample_batch = next(iter(val_loader))
    src_tokens = sample_batch['source'][:2]  # take 2 samples
    greedy_out = model.generate(src_tokens, max_length=12, start_token=1, end_token=2)
    beam_out = model.beam_search(src_tokens, max_length=12, beam_size=4, start_token=1, end_token=2)
    print("Greedy:", greedy_out)
    print("Beam:", beam_out)


if __name__ == "__main__":
    main()


