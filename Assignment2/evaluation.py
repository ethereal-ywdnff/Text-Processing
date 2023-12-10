import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def macro_f1_score(real_sentiment, pred_sentiment, classes):
    """
    Calculate the macro F1 score.

    Args:
    real_sentiment (list of int): True class labels.
    pred_sentiment (list of int): Predicted class labels.
    classes (list of int): List of unique class labels.

    Returns:
    float: The macro F1 score.
    """
    f1_scores = []

    for i in classes:
        tp = sum((real_sentiment == i) & (pred_sentiment == i))
        fp = sum((real_sentiment != i) & (pred_sentiment == i))
        fn = sum((real_sentiment == i) & (pred_sentiment != i))

        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

        f1_scores.append(f1)

    return np.mean(f1_scores)


def generate_confusion_matrix(real_sentiment, pred_sentiment, classes):
    """
    Generate a confusion matrix.

    Args:
    real_sentiment (list of int): True class labels.
    y_pred (list of int): Predicted class labels.
    pred_sentiment (list of int): List of unique class labels.

    Returns:
    array: Confusion matrix.
    """
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    matrix_size = len(classes)
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    for true, pred in zip(real_sentiment, pred_sentiment):
        if true in class_to_index and pred in class_to_index:
            matrix[class_to_index[true]][class_to_index[pred]] += 1

    return matrix


def plot_confusion_matrix(matrix, classes):
    """
    Plots a confusion matrix using Seaborn's heatmap.

    Args:
    matrix (array): The confusion matrix to be plotted.
    class (list of str): The labels for the classes.
    """
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Real Sentiment')
    plt.xlabel('Predicted Sentiment')
    plt.title('Confusion Matrix')
    plt.show()

