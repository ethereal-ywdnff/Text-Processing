import numpy as np
import math

class NaiveBayesClassifier:
    def __init__(self):
        self.smoothing_factor = 0.8
        self.log_class_priors = {}
        self.word_counts = None
        self.vocab = None
        self.n_classes = None  # Store the different types of sentiment present in the training data

    def fit(self, train_data, sentiment):
        """
        Fit the Naive Bayes classifier according to train_data, sentiment.

        Args:
        train_data (list of str)
        sentiment (list of int)
        """
        self.n_classes = np.unique(sentiment)
        if len(self.n_classes) == 3:
            self.smoothing_factor = 0.5
        self.word_counts = {c: {} for c in self.n_classes}
        self.vocab = set()
        n_samples = len(sentiment)

        for c in self.n_classes:
            data = [x for x, label in zip(train_data, sentiment) if label == c]
            count = len(data)
            self.log_class_priors[c] = np.log(count / n_samples)

            for text in data:
                for word in text:
                    if word not in self.word_counts[c]:
                        self.word_counts[c][word] = 0
                    self.word_counts[c][word] += 1
                    self.vocab.add(word)

        for c in self.n_classes:
            total_count = sum(self.word_counts[c].values())
            for word in self.vocab:
                self.word_counts[c][word] = np.log(
                    (self.word_counts[c].get(word, 0.0) + self.smoothing_factor) /
                    (total_count + len(self.vocab) * self.smoothing_factor)
                )

    def predict(self, data):
        """
        Perform classification on an array of test vectors X.

        Args:
        data (list of str): The input samples

        Returns:
        list of int: The predicted Sentiment for each sample in data.
        """
        results = []
        for text in data:
            class_scores = {c: self.log_class_priors[c] for c in self.n_classes}
            for word in text:
                if word in self.vocab:
                    for c in self.n_classes:
                        class_scores[c] += self.word_counts[c].get(word, 0)

            results.append(max(class_scores, key=class_scores.get))

        return results

    def compute_tfidf(self, documents):
        """
        Compute the TF-IDF score for each word in each document.

        Args:
        documents (list of list of str): List of documents with each document as a list of words.

        Returns:
        dict: A dictionary where keys are document indices and values are dictionaries of word TF-IDF scores.
        """
        word_doc_freq = {}
        # total_docs = 0
        tfidf_scores = {}
        total_docs = len(documents)
        for i, doc in enumerate(documents):
            word_freq = {}
            for word in doc:
                word_freq[word] = word_freq.get(word, 0) + 1

            for word in word_freq:
                word_doc_freq[word] = word_doc_freq.get(word, 0) + 1
            tfidf_scores[i] = {word: (word_freq[word] / len(doc)) * math.log(total_docs / word_doc_freq[word])
                               for word in word_freq}

        return tfidf_scores


# class NaiveBayesClassifier:
#     def __init__(self):
#         self.smoothing_factor = 0.5
#         self.word_doc_freq = {}
#         self.tfidf_scores = {}
#         self.class_prob = {}
#         self.word_class_prob = {}
#         self.vocab = set()
#         self.total_docs = 0
#         self.n_classes = None
#
#     def compute_tfidf(self, documents):
#         """
#         Compute the TF-IDF score for each word in each document.
#         """
#         self.total_docs = len(documents)
#         for doc_index, doc in enumerate(documents):
#             word_freq = {}
#             for word in doc:
#                 word_freq[word] = word_freq.get(word, 0) + 1
#                 self.vocab.add(word)
#                 self.word_doc_freq[word] = self.word_doc_freq.get(word, 0) + 1
#
#             doc_tfidf = {word: (freq / len(doc)) * math.log(self.total_docs / self.word_doc_freq[word])
#                          for word, freq in word_freq.items()}
#             self.tfidf_scores[doc_index] = doc_tfidf
#
#     def fit(self, X, y):
#         """
#         Fit the Naive Bayes classifier using TF-IDF scores.
#         """
#         self.compute_tfidf(X)
#         self.n_classes = np.unique(y)
#         for c in np.unique(y):
#             class_docs_indices = [i for i, label in enumerate(y) if label == c]
#             class_docs = [self.tfidf_scores[i] for i in class_docs_indices]
#             total_class_docs = len(class_docs)
#             self.class_prob[c] = total_class_docs / len(y)
#
#             word_scores = {}
#             for doc in class_docs:
#                 for word, score in doc.items():
#                     word_scores[word] = word_scores.get(word, 0) + score
#
#             total_score = sum(word_scores.values())
#             self.word_class_prob[c] = {word: (score + self.smoothing_factor) / (total_score + len(self.vocab)*self.smoothing_factor)
#                                        for word, score in word_scores.items()}
#
#     def predict(self, X):
#         """
#         Perform classification on an array of test vectors X.
#         """
#         self.compute_tfidf(X)
#
#         results = []
#         for doc_index, doc in enumerate(X):
#             class_scores = {c: math.log(self.class_prob[c]) for c in self.class_prob}
#             for word in doc:
#                 if word in self.vocab:
#                     for c in self.class_prob:
#                         class_scores[c] += math.log(self.word_class_prob[c].get(word, 1 / (sum(self.word_class_prob[c].values()) + len(self.vocab))))
#
#             results.append(max(class_scores, key=class_scores.get))
#
#         return results



