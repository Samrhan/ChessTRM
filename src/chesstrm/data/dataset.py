import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from chesstrm.data.mapping import move_to_index

class ChessDataset(Dataset):
    def __init__(self, h5_path: str, use_swmr: bool = True, in_memory: bool = False, target_col: str = 'target_d1_move'):
        """
        Args:
            h5_path: Path to the .h5 file.
            use_swmr: Whether to use Single Writer Multiple Reader mode.
            in_memory: If True, loads the entire dataset into RAM (be careful with size).
            target_col: Name of the dataset column to use as target (e.g. 'target_d1_move', 'target_d5_move', or 'y').
        """
        self.h5_path = h5_path
        self.use_swmr = use_swmr
        self.in_memory = in_memory
        self.target_col = target_col
        self.input_col = 'input_tensor'

        self.file = None
        self.x_data = None
        self.y_data = None

        # Open file once to get length and validate keys
        with h5py.File(self.h5_path, 'r', swmr=self.use_swmr) as f:
            # Determine input column name
            if 'input_tensor' in f:
                self.input_col = 'input_tensor'
            elif 'x' in f:
                self.input_col = 'x'
            else:
                raise ValueError(f"H5 file {h5_path} must contain 'input_tensor' or 'x'.")

            # Validate target column
            if self.target_col not in f:
                 # Fallback to 'y' if default wasn't found, or error
                if 'y' in f:
                    self.target_col = 'y'
                else:
                    raise ValueError(f"H5 file {h5_path} must contain '{self.target_col}' or 'y'.")

            self.length = len(f[self.input_col])

            if self.in_memory:
                print(f"Loading {self.length} samples into memory...")
                self.x_data = f[self.input_col][:]
                self.y_data = f[self.target_col][:]
                print("Done.")

    def _open_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r', swmr=self.use_swmr)
            self.x_dset = self.file[self.input_col]
            self.y_dset = self.file[self.target_col]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.in_memory:
            x = self.x_data[idx]
            y_raw = self.y_data[idx]
        else:
            self._open_file()
            x = self.x_dset[idx]
            y_raw = self.y_dset[idx]

        # Process x
        # x is expected to be (8, 8, 19) or (19, 8, 8)
        # We need (19, 8, 8) for PyTorch
        if x.shape == (8, 8, 19):
             x = np.transpose(x, (2, 0, 1))

        # Convert to float32 tensor
        x_tensor = torch.from_numpy(x).float()

        # Process y
        # y could be an index (int) or a UCI string (bytes/str)
        if isinstance(y_raw, (bytes, str, np.str_, np.bytes_)):
            if isinstance(y_raw, bytes):
                move_str = y_raw.decode('utf-8')
            else:
                move_str = str(y_raw)

            target_idx = move_to_index(move_str)
            if target_idx is None:
                # Fallback or error?
                # For now, raise error to catch data issues
                raise ValueError(f"Unknown move in dataset: {move_str}")
        else:
            # Assuming it's already an index
            target_idx = int(y_raw)

        return x_tensor, torch.tensor(target_idx, dtype=torch.long)

    def __del__(self):
        if self.file is not None:
            self.file.close()
