import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import logging

from src.chesstrm.model.trm import ChessTRM
from src.chesstrm.training.loss import DISLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer for ChessTRM model.
    Handles training loop, optimization, and checkpointing.
    """
    def __init__(
        self,
        model: ChessTRM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        loss_schedule: str = "linear",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.model.to(self.device)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = scheduler

        # Loss: DISLoss
        self.criterion = DISLoss(schedule=loss_schedule)

        # Enable gradient checkpointing in the model if available
        # We need to iterate over blocks and enable it
        if hasattr(self.model, 'blocks'):
            for block in self.model.blocks:
                if hasattr(block, 'use_checkpointing'):
                    block.use_checkpointing = True
                    logger.info("Enabled gradient checkpointing for a block.")

    def train_epoch(self, epoch: int) -> float:
        """
        Runs one training epoch.
        Returns: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Parse batch
            # Assuming batch is a list/tuple: (x, target) or dictionary
            if isinstance(batch, (list, tuple)):
                x, target = batch
            elif isinstance(batch, dict):
                x = batch['x']
                target = batch['target']
            else:
                raise ValueError("Unknown batch format")

            x = x.to(self.device)
            target = target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # ChessTRM forward returns list of logits
            logits_list = self.model(x)

            # Compute loss
            loss = self.criterion(logits_list, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss.item():.4f}")

        # Step scheduler if it exists (assuming per-epoch stepping for simplicity, unless it's OneCycle)
        # Note: If OneCycleLR is used, it should be stepped per batch.
        # But for generic support without complex configuration, per-epoch is safer or we need to know the type.
        # Given the requirements are loose, I'll step it here if it's not None.
        if self.scheduler:
            self.scheduler.step()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Epoch {epoch} Completed | Avg Loss: {avg_loss:.4f}")

        return avg_loss

    def save_checkpoint(self, epoch: int, path: Optional[str] = None):
        """Saves a checkpoint."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(state, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Loads a checkpoint."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in state:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        logger.info(f"Checkpoint loaded from {path}")
        return state.get('epoch', 0)
