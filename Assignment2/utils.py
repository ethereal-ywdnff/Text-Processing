import re

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

def preprocess(text):
    """
        Preprocess the input text for sentiment analysis.

        Args:
        text (str): Text to be preprocessed.

        Returns:
        str: Preprocessed text.
        """

    # Convert to lowercase
    text = text.lower()
    # text = re.sub(r'[^\w\s]', '', text)

    # Tokenization
    text = word_tokenize(text)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(token) for token in text]

    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    text = [token for token in text if token not in stop_words and token.isalpha()]

    # Reconstruct the text
    # text = ' '.join(tokens)

    return text

def load_data(file_path):
    """
    Load the dataset from a given file path.

    Args:
    file_path (str): Path to the dataset file.

    Returns:
    DataFrame: Pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path, sep="\t", encoding='utf-8')

# def preprocess(text):
#     """
#     Preprocess the input text.
#
#     Args:
#     text (str): Text to be preprocessed.
#
#     Returns:
#     str: Preprocessed text.
#     """
#     # Convert to lowercase
#     text = text.lower()
#
#     # Remove punctuation
#     text = re.sub(r'[^\w\s]', '', text)
#
#     text = text.split()
#
#     return text


def map_labels_5_to_3(label):
    """
    Maps labels from a 5-value scale to a 3-value scale.

    Args:
    label (str): The label on a 5-value scale.

    Returns:
    int: The corresponding label on a 3-value scale.
    """

    if label == 0 or label == 1:
        return 0
    elif label == 3 or label == 4:
        return 2
    else:
        return 1


