import torch
import chess
import h5py
import numpy as np
import sys

def main():
    print(f"Hello from ChessTRM!")
    print(f"Python version: {sys.version}")
    print(f"Torch version: {torch.__version__}")
    print(f"Python-chess version: {chess.__version__}")
    print(f"H5py version: {h5py.__version__}")
    print(f"Numpy version: {np.__version__}")

    # Simple check for CUDA
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is NOT available")

if __name__ == "__main__":
    main()
