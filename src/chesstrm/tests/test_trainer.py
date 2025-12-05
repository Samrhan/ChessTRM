import torch
from torch.utils.data import DataLoader, TensorDataset
from src.chesstrm.model.trm import ChessTRM
from src.chesstrm.training.trainer import Trainer
import shutil
import os
import torch.optim as optim

def test_trainer_integration():
    """Test full training loop for one epoch with dummy data."""
    # Setup
    batch_size = 4
    num_batches = 5
    d_model = 64 # Small model for test
    n_heads = 4
    n_layers = 1

    # Create dummy data
    # Input: (Batch, 19, 8, 8)
    x_data = torch.randn(batch_size * num_batches, 19, 8, 8)
    # Target: (Batch,) indices in [0, 1968)
    y_data = torch.randint(0, 1968, (batch_size * num_batches,))

    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ChessTRM(d_model=d_model, n_heads=n_heads, n_layers=n_layers)

    # Initialize Trainer
    checkpoint_dir = "test_checkpoints"
    trainer = Trainer(
        model=model,
        train_loader=dataloader,
        device="cpu", # Use CPU for test stability/availability
        checkpoint_dir=checkpoint_dir
    )

    # Run 1 epoch
    avg_loss = trainer.train_epoch(epoch=1)

    assert avg_loss > 0
    assert os.path.exists(checkpoint_dir)

    # Test checkpoint saving
    trainer.save_checkpoint(epoch=1)
    expected_ckpt = os.path.join(checkpoint_dir, "checkpoint_epoch_1.pt")
    assert os.path.exists(expected_ckpt)

    # Cleanup
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

def test_trainer_checkpointing_enable():
    """Verify that Trainer enables checkpointing on model blocks."""
    model = ChessTRM(d_model=32, n_heads=2, n_layers=1)

    # Verify initially False (default in code I saw)
    for block in model.blocks:
        block.use_checkpointing = False

    trainer = Trainer(
        model=model,
        train_loader=DataLoader(TensorDataset(torch.randn(1,19,8,8), torch.randint(0,10,(1,)))),
        device="cpu"
    )

    # Verify enabled
    for block in model.blocks:
        assert block.use_checkpointing is True

def test_trainer_with_scheduler():
    """Test that scheduler steps."""
    model = ChessTRM(d_model=32, n_heads=2, n_layers=1)

    trainer = Trainer(
        model=model,
        train_loader=DataLoader(TensorDataset(torch.randn(1,19,8,8), torch.randint(0,10,(1,)))),
        device="cpu"
    )

    # Add scheduler manually for test (usually done in init but optimizer is created there)
    optimizer = trainer.optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    trainer.scheduler = scheduler

    initial_lr = optimizer.param_groups[0]['lr']
    trainer.train_epoch(epoch=1)
    final_lr = optimizer.param_groups[0]['lr']

    assert final_lr < initial_lr, f"LR should have decreased. Initial: {initial_lr}, Final: {final_lr}"
