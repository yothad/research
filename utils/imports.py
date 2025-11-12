import pandas as pd

def import_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Imports a CSV file into a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.
    - **kwargs: Additional keyword arguments to pass to pd.read_csv().

    Returns:
    - pd.DataFrame: The imported data as a DataFrame.
    """
    return pd.read_csv(file_path, **kwargs)


def import_excel(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Imports an excel file into a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.
    - **kwargs: Additional keyword arguments to pass to pd.read_excel().

    Returns:
    - pd.DataFrame: The imported data as a DataFrame.
    """
    return pd.read_excel(file_path, **kwargs)


def import_yml(file_path: str) -> dict:
    """
    Imports a YAML file and returns its contents as a dictionary.

    Parameters:
    - file_path (str): The path to the YAML file.

    Returns:
    - dict: The contents of the YAML file.
    """
    import yaml
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)