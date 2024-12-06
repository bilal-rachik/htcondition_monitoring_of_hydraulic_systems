from pathlib import Path
from src.utils import load_dataset
import pandas as pd

def test_load_dataset():
    path_dir = Path(__file__).parent.joinpath("input_data")
    file_names = ["FS1.txt", "PS2.txt"]
    data_frames = load_dataset(path_dir, file_names )
    assert isinstance(data_frames, pd.DataFrame)
    assert len(data_frames) == 2




