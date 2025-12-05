import h5py
import argparse
import numpy as np

def inspect(file_path):
    print(f"Inspecting {file_path}...")
    try:
        with h5py.File(file_path, 'r') as f:
            print("Keys:", list(f.keys()))

            for key in f.keys():
                dset = f[key]
                print(f"\nDataset: {key}")
                print(f"  Shape: {dset.shape}")
                print(f"  Dtype: {dset.dtype}")

                # Check for NaNs if numeric
                if np.issubdtype(dset.dtype, np.number):
                    # Check first chunk
                    sample = dset[:min(100, len(dset))]
                    if np.isnan(sample).any():
                        print("  WARNING: NaNs detected in first 100 samples!")
                    else:
                        print("  No NaNs in first 100 samples.")

                # If it's y, show some examples
                if key == 'y' or key == 'moves':
                    sample = dset[:min(5, len(dset))]
                    print(f"  First 5 samples: {sample}")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to H5 file")
    args = parser.parse_args()
    inspect(args.file)
