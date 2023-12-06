from nltk import pos_tag

def select_words(text):
    """
    Extracts the adjectives, nouns, verbs, adverbs and interrogative pronouns.

    Args:
    text (str): The data need to be extracted features.

    Returns:
    list of str: A list of adjectives and nouns from the document.
    """
    pos_tags = pos_tag(text)
    # JJ: adjective  NN: noun   VB: verb   RB: adverb   WP: interrogative pronoun
    adj_nouns = [word for word, tag in pos_tags if tag.startswith('JJ') or tag.startswith('NN')
                 or tag.startswith('VB') or tag.startswith('RB') or tag.startswith('WP')]
    return adj_nouns


def feature_selection(text):
    """
    Applies feature selection on the text based on the lexical properties of the words.

    Args:
    text (list of str): data of 'Phrase'.

    Returns:
    list of str: The transformed text.
    """
    selected_corpus = []
    for doc in text:
        selected_corpus.append(select_words(doc))
    return selected_corpus










