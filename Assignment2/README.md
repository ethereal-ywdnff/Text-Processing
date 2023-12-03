# Assignment2: Sentiment Analysis of Movie Reviews
### Overview
This project implements a Naive Bayes classifier for sentiment analysis of movie reviews that either employ feature selection or not. The classifier is built from scratch and can handle different scales of sentiment (e.g., 3-class or 5-class). Then macro-F1 score is used to evaluate it. Four files with the predictions and confusion matrix can be generated if needed.


### Requirements
- Python version: 3.9.x or above
- Libraries used:
  - numpy (for numerical operations)
  - pandas (for data handling)
  - seaborn to render the confusion matrix
  - matplotlib to plot the confusion matrix
  - nltk to preprocess the 'Phrase'


### Installation
No additional installation is required for standard Python libraries like numpy and pandas. But seaborn and nltk are required to be installed.

```bash
pip install seaborn
```

```bash
pip install nltk
```


### Files Description
- NB_sentiment_analyser.py: The main part to run the sentiment analysis.
- naive_bayes.py: Implementation of the Naive Bayes classifier.
- feature_selection.py: Module for feature selection in the dataset.
- evaluation.py: Module for evaluating the classifier (macro F1 score and confusion matrix).
- utils.py: Utility functions used to implement the project (e.g. loading and preprocessing the data).


### Running the System

1. ``cd`` into the right folder.

2. Run the model:
- Results for ***5*** classes and using all ***words*** as feature
```bash
python NB_sentiment_analyser.py train.tsv dev.tsv test.tsv -classes 5 -features all_words
```

- Results for ***5*** classes and using ***features*** as feature
```bash
python NB_sentiment_analyser.py train.tsv dev.tsv test.tsv -classes 5 -features features
```

- Results for ***3*** classes and using ***all words*** as feature
```bash
python NB_sentiment_analyser.py train.tsv dev.tsv test.tsv -classes 3 -features all_words
```

- Results for ***3*** classes and using ***features*** as feature
```bash
python NB_sentiment_analyser.py train.tsv dev.tsv test.tsv -classes 3 -features features
```

3. Output files and confusion matrix:
add -output_files -confusion_matrix to above command line, e.g.
```bash
python NB_sentiment_analyser.py train.tsv dev.tsv test.tsv -classes 5 -features all_words -output_files -confusion_matrix
```


