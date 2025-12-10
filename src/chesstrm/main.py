import argparse
import logging
import sys
import os
import torch
from torch.utils.data import DataLoader
from chesstrm.model.trm import ChessTRM
from chesstrm.training.trainer import Trainer
from chesstrm.data.dataset import ChessDataset
from chesstrm.inference import load_model, predict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ChessTRM")

def train(args):
    logger.info(f"Starting training with args: {args}")

    # Check device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    logger.info(f"Using device: {device}")

    # Dataset
    if not args.data:
        logger.error("Data path (--data) is required for training.")
        return

    logger.info(f"Loading dataset from {args.data}...")
    try:
        dataset = ChessDataset(args.data, use_swmr=True, in_memory=args.in_memory)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True if device == "cuda" else False
    )

    # Model
    model = ChessTRM(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_recursion=args.max_recursion
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        loss_schedule=args.loss_schedule
    )

    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}...")
        try:
             start_epoch = trainer.load_checkpoint(args.resume) + 1
             logger.info(f"Resumed at epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Failed to resume checkpoint: {e}")
            return

    # Loop
    for epoch in range(start_epoch, args.epochs):
        avg_loss = trainer.train_epoch(epoch)
        if (epoch + 1) % args.save_interval == 0:
            trainer.save_checkpoint(epoch)

    # Save final
    trainer.save_checkpoint(args.epochs, os.path.join(args.checkpoint_dir, "final.pt"))
    logger.info("Training complete.")

def run_inference(args):
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    if not args.fen:
        logger.error("FEN string (--fen) is required for inference.")
        return

    model = load_model(args.model_path, device=device)

    logger.info(f"Analyzing position: {args.fen}")
    top_moves = predict(args.fen, model, top_k=args.top_k, n_steps=args.n_steps, device=device)

    print("\nTop Moves:")
    for move, prob in top_moves:
        print(f"{move}: {prob:.4f}")

def main():
    parser = argparse.ArgumentParser(description="ChessTRM: Tiny Recursive Model for Chess")
    subparsers = parser.add_subparsers(dest="mode", help="Mode: train or inference")

    # Train Parser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", type=str, required=True, help="Path to H5 dataset")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    train_parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    train_parser.add_argument("--save-interval", type=int, default=1, help="Save checkpoint every N epochs")
    train_parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers")
    train_parser.add_argument("--in-memory", action="store_true", help="Load dataset into RAM")
    train_parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")

    # Model Hparams
    train_parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    train_parser.add_argument("--n-heads", type=int, default=8, help="Number of heads")
    train_parser.add_argument("--n-layers", type=int, default=2, help="Number of layers per block")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    train_parser.add_argument("--max-recursion", type=int, default=16, help="Max recursion steps")
    train_parser.add_argument("--loss-schedule", type=str, default="linear", choices=["linear", "uniform"], help="DIS Loss schedule")

    # Inference Parser
    inf_parser = subparsers.add_parser("inference", help="Run inference")
    inf_parser.add_argument("--fen", type=str, required=True, help="FEN string")
    inf_parser.add_argument("--model-path", type=str, help="Path to trained model checkpoint")
    inf_parser.add_argument("--top-k", type=int, default=5, help="Number of top moves to show")
    inf_parser.add_argument("--n-steps", type=int, default=16, help="Recursion steps for inference")
    inf_parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "inference":
        run_inference(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
