import pandas as pd
from pathlib import Path
from typing import List

def load_dataset(file_path: Path, file_names: List[str]) -> pd.DataFrame:
    """Load multiple data files and concatenate them into a single DataFrame."""
    data_frames = [pd.read_csv(file_path.joinpath(file_name), sep='\t', header=None) for file_name in file_names]
    return pd.concat(data_frames, axis=1)