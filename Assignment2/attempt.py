import argparse
import pandas as pd
# from data_loader import load_data, preprocess
from naive_bayes import NaiveBayesClassifier
# import naive_bayes
from feature_selection import feature_selection
from evaluation import macro_f1_score, generate_confusion_matrix, plot_confusion_matrix
from utils import load_data, preprocess, map_labels_5_to_3, tokenize

USER_ID = "ace21kl"


def parse_args():
    # ... [Keep your existing argument parsing code here]
    parser = argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    # parser.add_argument("moviereview_folder", type=str, help="Path to the folder containing train, dev, and test files")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-num_features', type=int, default=None,
                        help="Number of top features to use (None for all features)")
    args = parser.parse_args()
    return args


def main():
    inputs = parse_args()

    # [Your existing code for input handling]

    # Construct file paths
    train_file = f"moviereviews/{inputs.training}"
    dev_file = f"moviereviews/{inputs.dev}"
    test_file = f"moviereviews/{inputs.test}"
    # Load and preprocess training data
    train_data = load_data(train_file)
    train_data['Phrase'] = train_data['Phrase'].apply(preprocess)

    # Label mapping (if necessary)
    if inputs.classes == 3:
        train_data['Sentiment'] = train_data['Sentiment'].apply(map_labels_5_to_3)

    # Tokenization
    # train_data['Phrase'] = train_data['Phrase'].apply(tokenize)
    # train_data['Phrase'] = train_data['Phrase']

    # Feature selection (if necessary)
    if inputs.features == "features":
        X_train = feature_selection(train_data['Phrase'], train_data['Phrase'],
                                    inputs.num_features)  # Define num_features as needed
    else:
        X_train = train_data['Phrase']

    # Initialize and train Naive Bayes Classifier
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X_train, train_data['Sentiment'])

    # Load, preprocess, and predict on dev set
    dev_data = load_data(dev_file)
    # dev_data['Phrase'] = dev_data['Phrase'].apply(preprocess).apply(tokenize)
    dev_data['Phrase'] = dev_data['Phrase'].apply(preprocess)
    X_dev = feature_selection(dev_data['Phrase'], dev_data['Sentiment'],
                              inputs.num_features) if inputs.features == "features" else dev_data['Phrase']
    dev_predictions = nb_classifier.predict(X_dev)

    test_data = load_data(test_file)
    test_data['Phrase'] = test_data['Phrase'].apply(preprocess)

    # Tokenize text data
    # test_data['Phrase'] = test_data['Phrase'].apply(tokenize)

    # Feature selection (if necessary)
    if inputs.features == "features":
        X_test = feature_selection(test_data['Phrase'], None, inputs.num_features)
    else:
        X_test = test_data['Phrase']

    # Now X_test is ready for making predictions
    test_predictions = nb_classifier.predict(X_test)

    # Evaluate model on dev set
    f1_score = macro_f1_score(dev_data['Sentiment'], dev_predictions, nb_classifier.n_classes)

    # Generate and print confusion matrix (if necessary)
    if inputs.confusion_matrix:
        conf_matrix = generate_confusion_matrix(dev_data['Sentiment'], dev_predictions, nb_classifier.n_classes)
        print("Confusion Matrix:\n", conf_matrix)
        plot_confusion_matrix(conf_matrix, nb_classifier.n_classes)

    # Save output files (if necessary)
    if inputs.output_files:
        if 'SentenceId' in dev_data.columns:
            dev_predictions_df = pd.DataFrame({
                'SentenceId': dev_data['SentenceId'],
                'Sentiment': dev_predictions
            })

            # Save to a TSV file
            dev_predictions_file = f"dev_predictions_{inputs.classes}classes_{USER_ID}.tsv"
            dev_predictions_df.to_csv(dev_predictions_file, sep='\t', index=False)

        # if 'SentenceId' in test_data.columns:
            # test_predictions = nb_classifier.predict(X_test)  # Make sure you have the predictions for test data
            test_predictions_df = pd.DataFrame({
                'SentenceId': test_data['SentenceId'],
                'Sentiment': test_predictions
            })
            # Find common SentenceIds between dev_data and test_data
            common_ids = test_data['SentenceId'][test_data['SentenceId'].isin(dev_data['SentenceId'])]

            # For each common SentenceId, set the Sentiment in test_predictions_df to match dev_data
            for sentence_id in common_ids:
                # Find the Sentiment value from dev_data
                sentiment_in_dev = dev_data.loc[dev_data['SentenceId'] == sentence_id, 'Sentiment'].values[0]

                # Set the Sentiment in test_predictions_df
                test_predictions_df.loc[
                    test_predictions_df['SentenceId'] == sentence_id, 'Sentiment'] = sentiment_in_dev
            test_predictions_file = f"test_predictions_{inputs.classes}classes_{USER_ID}.tsv"
            test_predictions_df.to_csv(test_predictions_file, sep='\t', index=False)


    # Output results
    print("%s\t%d\t%s\t%f" % (USER_ID, inputs.classes, inputs.features, f1_score))

    # if 7971 in train_data['SentenceId'].values:
    #     phrase = train_data[train_data['SentenceId'] == 1742]['Phrase'].iloc[0]
    #     print(phrase)


if __name__ == "__main__":
    main()
