import numpy as np

def compute_tf(document):
    """
    Compute term frequency for each term in a document.

    Args:
    document (list of str): The document represented as a list of words.

    Returns:
    dict: A dictionary where keys are words and values are their term frequency.
    """
    tf = {}
    for word in document:
        tf[word] = tf.get(word, 0) + 1
    total_words = len(document)
    return {word: count / total_words for word, count in tf.items()}

def compute_idf(corpus):
    """
    Compute inverse document frequency for each unique term in the corpus.

    Args:
    corpus (list of list of str): The corpus where each document is a list of words.

    Returns:
    dict: A dictionary where keys are words and values are their inverse document frequency.
    """
    idf = {}
    total_docs = len(corpus)
    for document in corpus:
        unique_words = set(document)
        for word in unique_words:
            idf[word] = idf.get(word, 0) + 1
    return {word: np.log(total_docs / df) for word, df in idf.items()}

def compute_tfidf(corpus):
    """
    Compute TF-IDF score for each term in the corpus.

    Args:
    corpus (list of list of str): The corpus where each document is a list of words.

    Returns:
    dict: A dictionary where keys are words and values are their TF-IDF score.
    """
    idf = compute_idf(corpus)
    tfidf = {word: {} for word in set(word for doc in corpus for word in doc)}
    for idx, document in enumerate(corpus):
        tf = compute_tf(document)
        for word, tf_val in tf.items():
            tfidf[word][idx] = tf_val * idf[word]
    return tfidf


def feature_selection(X):
    """
    Applies TF-IDF based feature selection to the dataset.

    Args:
    X (list of list of str): The input samples, each a list of words.
    n (int): Number of top words to select.

    Returns:
    list of list of str: The transformed input samples with only the top N words.
    """
    tfidf_scores = compute_tfidf(X)
    word_scores = {word: sum(docs.values()) for word, docs in tfidf_scores.items()}
    sorted_words = sorted(word_scores, key=word_scores.get, reverse=True)
    X_selected = [[word for word in document if word in sorted_words] for document in X]
    return X_selected







