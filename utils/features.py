import pandas as pd
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


def target_encoding_column(df: pd.DataFrame, column_name: str, target_name: str) -> pd.DataFrame:
    # groupby by column and calc target ratio
    freq_column = f'{column_name}_target_freq'
    gb_df = df.groupby(column_name, as_index=False).agg(counter=(target_name, 'count'),
                                                        target_sum=(target_name, 'sum'),
                                                        )
    gb_df[freq_column] = gb_df['target_sum'] / gb_df['counter'].sum()
    gb_df = gb_df.drop(columns={['target_sum', 'counter']})

    # merge target ratio to original df
    df = df.merge(gb_df, how='inner', on=column_name)
    return df, freq_column


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

