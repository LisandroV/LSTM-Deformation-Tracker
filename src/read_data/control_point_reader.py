import pandas as pd


def read_control_points_file(file_path: str):
    data = pd.read_csv(file_path)
    return pd.DataFrame(data)
