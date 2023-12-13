# Assignment2: Sentiment Analysis of Movie Reviews
### Overview
This project implements a Naive Bayes classifier for sentiment analysis of movie reviews that either employ feature selection or not. The classifier is built from scratch and can handle different scales of sentiment (e.g., 3-class or 5-class). Then macro-F1 score is used to evaluate it. Four files with the predictions and confusion matrix can be generated if needed.


### Files Description
- NB_sentiment_analyser.py: The main part to run the sentiment analysis.
- naive_bayes.py: Implementation of the Naive Bayes classifier.
- feature_selection.py: Module for feature selection in the dataset.
- evaluation.py: Module for evaluating the classifier (macro F1 score and confusion matrix).
- utils.py: Utility functions used to implement the project (e.g. loading and preprocessing the data).


### Requirements
- Python version: 3.9.x or above
- Libraries used:
  - numpy (for numerical operations)
  - pandas (for data handling)
  - seaborn to render the confusion matrix
  - matplotlib to plot the confusion matrix
  - nltk to preprocess the 'Phrase'
- Installation
  - There is a file named pip ``requirements.txt``, which can be used to install all the needed libraries.
  - Below is the command line to install the libraries.
```bash
pip install -r requirements.txt
```
