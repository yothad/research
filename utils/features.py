import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction import FeatureHasher


def frequency_encoding_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    # groupby by column and calc target ratio
    freq_column = f'{column_name}_freq'
    gb_df = df.groupby(column_name).size().reset_index()
    gb_df[freq_column] = gb_df[0] / gb_df[0].sum()
    gb_df = gb_df.drop(columns={0})

    # merge target ratio to original df
    df = df.merge(gb_df, how='inner', on=column_name)
    return df, freq_column


def target_encoding_kfold(
    df: pd.DataFrame, 
    column_name: str, 
    target_name: str, 
    n_splits: int = 5, 
    alpha: float = 1.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perform K-fold target encoding for a categorical column.

    Parameters:
    - df: input dataframe
    - column_name: categorical column to encode
    - target_name: target column
    - n_splits: number of folds
    - alpha: smoothing factor
    - random_state: random seed for reproducibility

    Returns:
    - df with new column "<column_name>_te"
    """
    df = df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    new_col = f"{column_name}_te"
    df[new_col] = np.nan
    global_mean = df[target_name].mean()

    for train_idx, val_idx in kf.split(df):
        train, val = df.iloc[train_idx], df.iloc[val_idx]

        # Compute target mean and count per category in training fold
        stats = train.groupby(column_name)[target_name].agg(['mean', 'count']).reset_index()
        
        # Apply smoothing
        stats['smoothed'] = (stats['mean'] * stats['count'] + global_mean * alpha) / (stats['count'] + alpha)
        
        # Map to validation fold
        mapping = dict(zip(stats[column_name], stats['smoothed']))
        df.loc[val_idx, new_col] = df.loc[val_idx, column_name].map(mapping).fillna(global_mean)

    return df


def one_hot_encoding_column(df: pd.DataFrame, column_name: str):
    column_list = []
    for val in df[column_name].unique():
        df.loc[df[column_name] == val, f"{column_name}_{val}_hotfix"] = 1
        df[f"{column_name}_{val}_hotfix"] = df[f"{column_name}_{val}_hotfix"].fillna(0)
        column_list.append(f"{column_name}_{val}_hotfix")

    return df, column_list


def hash_column(df: pd.DataFrame, column_name: str, n_features: int = 32) -> pd.DataFrame:
    df[column_name] = df[column_name].astype(str)
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    hashed_features = hasher.transform(df[column_name].apply(lambda x: [x]))

    # Convert sparse matrix to DataFrame
    hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f"{column_name}_hash_{i}" for i in range(n_features)])
    hashed_columns = hashed_df.columns.tolist()

    # Concatenate hashed features to original DataFrame
    df = pd.concat([df.reset_index(drop=True), hashed_df.reset_index(drop=True)], axis=1)
    return df, hashed_columns


def extract_ip_prefix(df, column_name, num_levels=1):
    df[f'IP_prefix_{num_levels}'] = df[column_name].astype(str).apply(lambda x: '.'.join(x.split('.')[:num_levels]))
    return df

