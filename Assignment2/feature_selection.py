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

def select_top_n_words(tfidf_scores, n):
    """
    Selects the top N words with the highest TF-IDF scores.

    Args:
    tfidf_scores (dict of dict): TF-IDF scores for each term.
    n (int): Number of top words to select.

    Returns:
    set: A set of the top N words.
    """
    word_scores = {word: sum(docs.values()) for word, docs in tfidf_scores.items()}
    sorted_words = sorted(word_scores, key=word_scores.get, reverse=True)
    return set(sorted_words[:n])

def feature_selection(X, y, n):
    """
    Applies TF-IDF based feature selection to the dataset.

    Args:
    X (list of list of str): The input samples, each a list of words.
    n (int): Number of top words to select.

    Returns:
    list of list of str: The transformed input samples with only the top N words.
    """
    tfidf_scores = compute_tfidf(X)
    top_words = select_top_n_words(tfidf_scores, n)
    X_selected = [[word for word in document if word in top_words] for document in X]
    return X_selected


# def top_n_words(corpus, n=None):
#     """
#     Selects the top N most frequent words in the corpus.
#
#     Args:
#     corpus (list of list of str): The corpus of documents, where each document is represented as a list of words.
#     n (int, optional): Number of top words to select. If None, selects all words.
#
#     Returns:
#     set: A set of the top N words.
#     """
#     word_counts = {}
#     for document in corpus:
#         for word in document:
#             if word not in word_counts:
#                 word_counts[word] = 0
#             word_counts[word] += 1
#
#     # Sort words by their counts in descending order
#     sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
#
#     if n is not None:
#         sorted_words = sorted_words[:n]
#
#     return set(sorted_words)
#
#
# def feature_selection(X, y, n=None):
#     """
#     Selects features (words) based on their frequency in the corpus.
#
#     Args:
#     X (list of list of str): The training input samples. Each sample is a list of words.
#     y (list of int): The target values (class labels).
#     n (int, optional): Number of top words to select. If None, selects all words.
#
#     Returns:
#     list of list of str: The transformed input samples with only the top N words.
#     """
#     top_words = top_n_words(X, n)
#
#     X_selected = []
#     for document in X:
#         doc_selected = [word for word in document if word in top_words]
#         X_selected.append(doc_selected)
#
#     return X_selected



