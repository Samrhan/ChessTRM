import torch
from typing import List, Tuple, Optional
from chesstrm.model.trm import ChessTRM
from chesstrm.data.fen_to_tensor import fen_to_tensor
from chesstrm.data.mapping import index_to_move, get_num_actions

def load_model(model_path: Optional[str] = None, device: str = 'cpu') -> ChessTRM:
    """
    Loads the ChessTRM model.
    If model_path is None, returns an initialized model (useful for testing).
    """
    # Initialize model with default hyperparameters (check if they match training)
    model = ChessTRM(d_model=256, n_heads=8, n_layers=2, dropout=0.0)
    model.to(device)
    model.eval()

    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        # Handle state dict (if it's wrapped in 'model_state_dict' or just raw)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Clean up prefix if trained with DDP or similar (e.g. 'module.')
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        print(f"Model loaded from {model_path}")
    else:
        print("Model initialized (untrained).")

    return model

def predict(
    fen: str,
    model: ChessTRM,
    top_k: int = 5,
    n_steps: int = 8,
    device: str = 'cpu'
) -> List[Tuple[str, float]]:
    """
    Runs inference on a FEN string.

    Args:
        fen: The FEN string of the position.
        model: The loaded ChessTRM model.
        top_k: Number of top moves to return.
        n_steps: Number of recursion steps.
        device: 'cpu' or 'cuda'.

    Returns:
        List of (move_uci, probability) tuples, sorted by probability descending.
    """
    # 1. Prepare Input
    tensor = fen_to_tensor(fen)
    tensor = tensor.unsqueeze(0) # Add batch dimension: (1, 19, 8, 8)
    tensor = tensor.to(device)

    # 2. Run Model
    with torch.no_grad():
        all_logits = model(tensor, n_steps=n_steps)
        # Use the logits from the final step
        final_logits = all_logits[-1] # (1, 1968)

    # 3. Process Output
    probs = torch.softmax(final_logits, dim=1).squeeze(0) # (1968,)

    # Get top K
    top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for p, idx in zip(top_probs, top_indices):
        move_str = index_to_move(idx.item())
        if move_str:
            results.append((move_str, p.item()))
        else:
            results.append(("Unknown", p.item()))

    return results

if __name__ == "__main__":
    import sys
    # Simple CLI usage: python inference.py <fen> <model_path>
    if len(sys.argv) > 1:
        fen = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(model_path, device)

        print(f"Analyzing Position: {fen}")
        top_moves = predict(fen, model, device=device)

        print("Top Moves:")
        for move, prob in top_moves:
            print(f"{move}: {prob:.4f}")
    else:
        print("Usage: python -m chesstrm.inference <fen> [model_path]")
