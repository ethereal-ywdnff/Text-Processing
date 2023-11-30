import pandas as pd
import re

def load_data(file_path):
    """
    Load the dataset from a given file path.

    Args:
    file_path (str): Path to the dataset file.

    Returns:
    DataFrame: Pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path, sep="\t", encoding='utf-8')

def preprocess(text):
    """
    Preprocess the input text.

    Args:
    text (str): Text to be preprocessed.

    Returns:
    str: Preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    text = text.split()

    return text


