import pandas as pd

def import_csv(file_path, **kwargs) -> pd.DataFrame:
    """
    Imports a CSV file (or several, concatenated) into a pandas DataFrame.

    Parameters:
    - file_path (str | list[str]): The path to the CSV file, or a list of paths
      to be read and concatenated in order (e.g. a large file split into parts
      to stay under a size limit — each part keeps its own header).
    - **kwargs: Additional keyword arguments to pass to pd.read_csv().

    Returns:
    - pd.DataFrame: The imported data as a DataFrame.
    """
    if isinstance(file_path, (list, tuple)):
        return pd.concat([pd.read_csv(p, **kwargs) for p in file_path], ignore_index=True)
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