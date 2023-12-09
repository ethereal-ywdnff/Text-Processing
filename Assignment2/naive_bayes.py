import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
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
        # self.log_class_priors = np.zeros(len(self.n_classes), dtype=np.float64)
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
                    (self.word_counts[c].get(word, 0.0) + 1) / (total_count + len(self.vocab))
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
