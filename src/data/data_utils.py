import os
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from src.entities import SplittingParams

def load_data_from_s3(file_path: str, output_folder_path: str):
    os.system(f"s3cmd get -r --force {file_path} {output_folder_path}")

def put_to_s3(file_path: str, s3_file_path: Optional[str]):
    os.system(f"s3cmd put {file_path} {s3_file_path}")


def read_data(path: str) -> pd.DataFrame:
    """
    Read dataset from csv file and retrun pandas dataframe
    """
    df = pd.read_csv(path)
    return df


def split_train_validate_data(
    df: pd.DataFrame, splitting_params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe of all data to train and validate parts
    """
    df_train, df_validate = train_test_split(
        df,
        test_size=splitting_params.validate_size,
        random_state=splitting_params.random_state,
    )
    return df_train, df_validate
