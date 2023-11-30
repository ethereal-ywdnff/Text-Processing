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


def map_labels_5_to_3(label):
    """
    Maps labels from a 5-value scale to a 3-value scale.

    Args:
    label (int or str): The label on a 5-value scale.

    Returns:
    int or str: The corresponding label on a 3-value scale.
    """
    # if label in ['negative', 'somewhat negative']:
    #     return 'negative'
    # elif label in ['somewhat positive', 'positive']:
    #     return 'positive'
    # else:
    #     return 'neutral'
    # if label in [0, 1]:
    if label == 0 or label == 1:
        return 0
    elif label == 3 or label == 4:
        return 2
    else:
        return 1


def tokenize(text):
    """
    Tokenizes a string into words.

    Args:
    text (str): The text to be tokenized.

    Returns:
    list of str: A list of words.
    """
    # This is a simple tokenization process; consider using more sophisticated methods if necessary
    return text.split()

# You can add more utility functions as needed for your project
