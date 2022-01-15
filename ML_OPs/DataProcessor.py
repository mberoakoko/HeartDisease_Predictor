import pandas as pd
from pandas import DataFrame
from pathlib import Path

from typing import List, Union

_data_path = Path("./Data/heart.csv")
_num_feats = ('age', 'cholesterol', 'resting_blood_pressure', 'max_heart_rate_achieved', 'st_depression',
              'num_major_vessels')
_bin_feats = ('sex', 'fasting_blood_sugar', 'exercise_induced_angina', 'target')

_nom_feats = ('chest_pain_type', 'rest_electrocardiographic', 'st_slope', 'thalassemia')


def load_data(path: Union[Path, str] = _data_path) -> DataFrame:
    df = pd.read_csv(path)
    df = df[df["ca"] < 4]
    df = df[df["thal"] > 0]
    df = df.rename(columns={'cp': 'chest_pain_type', 'trestbps': 'resting_blood_pressure',
                       'chol': 'cholesterol', 'fbs': 'fasting_blood_sugar',
                       'restecg': 'rest_electrocardiographic', 'thalach': 'max_heart_rate_achieved',
                       'exang': 'exercise_induced_angina', 'oldpeak': 'st_depression',
                       'slope': 'st_slope', 'ca': 'num_major_vessels', 'thal': 'thalassemia'}, errors="raise")
    return df


def get_numerical_features() -> List[str]:
    return list(_num_feats)


def get_binary_features() -> List[str]:
    return list(_bin_feats)


def get_categorical_features() -> List[str]:
    return list(_nom_feats + _bin_feats)
