import re

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Example list of English stopwords
# stopwords = set([
#     "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
#     "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
#     "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
#     "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
#     "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
#     "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
#     "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
#     "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
#     "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
#     "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
#     "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
#     "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
# ])
# def preprocess(text):
#     # Convert to lowercase
#     text = text.lower()
#
#     # Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))
#
#     # Remove stopwords
#     words = text.split()
#     words = [word for word in words if word not in stopwords]
#
#     # Reconstruct the text
#     text = ' '.join(words)
#     text = text.split()
#
#     return text

def preprocess(text):  # 5 0.336963
    """
        Preprocess the input text for sentiment analysis.

        Args:
        text (str): Text to be preprocessed.

        Returns:
        str: Preprocessed text.
        """

    # Convert to lowercase
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove stop words and punctuation (optional: keep punctuation if needed)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]

    # Reconstruct the text
    text = ' '.join(tokens)

    return tokens


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


