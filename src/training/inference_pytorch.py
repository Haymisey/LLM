"""
PyTorch Inference Script for Amharic-Oromiffa Translation
========================================================

This script loads a trained PyTorch model and performs inference.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Ensure `src` is on path when running from repo root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from training.train_pytorch import TransformerModel, PositionalEncoding


def load_model(checkpoint_path: str, src_vocab_size: int, tgt_vocab_size: int, 
               d_model: int = 512, n_heads: int = 8, d_ff: int = 2048,
               n_encoder_layers: int = 6, n_decoder_layers: int = 6):
    """Load a trained model from checkpoint."""
    
    # Create model
    model = TransformerModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return model


def translate_text(model: TransformerModel, src_text: str, src_tokenizer, tgt_tokenizer,
                  max_length: int = 100, temperature: float = 1.0):
    """Translate a single text."""
    
    model.eval()
    
    # Tokenize source
    src_ids = src_tokenizer.encode(src_text, add_special_tokens=True)
    src_tensor = torch.tensor([src_ids], dtype=torch.long)
    
    # Generate translation
    with torch.no_grad():
        generated_ids = model.generate(
            src_tensor,
            max_length=max_length,
            start_token=tgt_tokenizer.sos_token_id,
            end_token=tgt_tokenizer.eos_token_id,
            temperature=temperature
        )
    
    # Decode
    generated_text = tgt_tokenizer.decode(generated_ids[0].tolist())
    
    return generated_text


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='PyTorch Amharic-Oromiffa Translation Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--src_vocab_size', type=int, default=10000, help='Source vocabulary size')
    parser.add_argument('--tgt_vocab_size', type=int, default=10000, help='Target vocabulary size')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--n_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--n_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--text', type=str, help='Text to translate (optional)')
    
    args = parser.parse_args()
    
    print("🚀 Loading PyTorch Translation Model")
    print("=" * 50)
    
    # Load model
    model = load_model(
        args.checkpoint,
        args.src_vocab_size,
        args.tgt_vocab_size,
        args.d_model,
        args.n_heads,
        args.d_ff,
        args.n_encoder_layers,
        args.n_decoder_layers
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model loaded on device: {device}")
    
    # Interactive translation
    if args.text:
        # Load tokenizers (you'll need to implement this based on your dataset)
        print("Note: You need to load your tokenizers here")
        print(f"Input text: {args.text}")
        print("Translation: [Tokenizers not loaded]")
    else:
        print("\nEnter Amharic text to translate (or 'quit' to exit):")
        while True:
            try:
                text = input("Amharic: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if text:
                    # Note: You need to load your tokenizers here
                    print("Translation: [Tokenizers not loaded]")
                    print("To use this script, you need to load your trained tokenizers.")
            except KeyboardInterrupt:
                break
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
