import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def macro_f1_score(y_true, y_pred, classes):
    """
    Calculate the macro F1 score.

    Args:
    y_true (list of int): True class labels.
    y_pred (list of int): Predicted class labels.
    classes (list of int): List of unique class labels.

    Returns:
    float: The macro F1 score.
    """
    f1_scores = []

    for cls in classes:
        tp = sum((y_true == cls) & (y_pred == cls))
        fp = sum((y_true != cls) & (y_pred == cls))
        fn = sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

        f1_scores.append(f1)

    return np.mean(f1_scores)


def generate_confusion_matrix(y_true, y_pred, classes):
    """
    Generate a confusion matrix from scratch.

    Args:
    y_true (list of int): True class labels.
    y_pred (list of int): Predicted class labels.
    classes (list of int): List of unique class labels.

    Returns:
    array: Confusion matrix.
    """
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    matrix_size = len(classes)
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    for true, pred in zip(y_true, y_pred):
        if true in class_to_index and pred in class_to_index:
            matrix[class_to_index[true]][class_to_index[pred]] += 1

    return matrix


def plot_confusion_matrix(matrix, class_labels):
    """
    Plots a confusion matrix using Seaborn's heatmap.

    Args:
    matrix (array): The confusion matrix to be plotted.
    class_labels (list of str): The labels for the classes.
    """
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
