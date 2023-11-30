import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.log_class_priors = {}
        self.word_counts = None
        self.vocab = None
        self.n_classes = None

    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier according to X, y.

        Args:
        X (list of list of str): The training input samples. Each sample is a list of words.
        y (list of int): The target values (class labels).
        """
        self.n_classes = np.unique(y)
        # self.log_class_priors = np.zeros(len(self.n_classes), dtype=np.float64)
        self.word_counts = {c: {} for c in self.n_classes}
        self.vocab = set()
        n_samples = len(y)

        for c in self.n_classes:
            X_c = [x for x, label in zip(X, y) if label == c]
            count = len(X_c)
            self.log_class_priors[c] = np.log(count / n_samples)

            for text in X_c:
                for word in text:
                    if word not in self.word_counts[c]:
                        self.word_counts[c][word] = 0
                    self.word_counts[c][word] += 1
                    self.vocab.add(word)

        for c in self.n_classes:
            total_count = sum(self.word_counts[c].values())
            for word in self.vocab:
                self.word_counts[c][word] = np.log(
                    (self.word_counts[c].get(word, 0.0) + 1) / (total_count + len(self.vocab))
                )

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Args:
        X (list of list of str): The input samples. Each sample is a list of words.

        Returns:
        list of int: The predicted class label for each sample in X.
        """
        results = []
        for text in X:
            class_scores = {c: self.log_class_priors[c] for c in self.n_classes}
            for word in text:
                if word in self.vocab:
                    for c in self.n_classes:
                        class_scores[c] += self.word_counts[c].get(word, 0)

            results.append(max(class_scores, key=class_scores.get))

        return results
