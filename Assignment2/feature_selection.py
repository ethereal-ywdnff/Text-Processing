from nltk import pos_tag


def select_words(text, cl):
    """
    Extracts the adjectives, nouns, verbs, adverbs and interrogative pronouns.

    Args:
    text (str): The data need to be extracted features.
    cl: 3-class oe 5-class

    Returns:
    list of str: text with specific features.
    """
    pos_tags = pos_tag(text)
    # JJ: adjective  NN: noun   VB: verb   RB: adverb   WP: interrogative pronoun
    # text = [word for word, tag in pos_tags if tag.startswith('JJ') or tag.startswith('NN')
    #              or tag.startswith('VB') or tag.startswith('RB')) or tag.startswith('WP')]
    # JJ: adjective  NN: noun, VB: verb, RB: adverb, CC: conjunction WP: interrogative pronoun, FW: foreign words,
    # IN: prepositions/subordinating conjunction, DT: determiners, CD: numerals, RP: e.g. "up", "off", "out"
    if cl == 5:
        text = [word for word, tag in pos_tags if tag.startswith('JJ') or tag.startswith('NN')
                or tag.startswith('VB') or tag.startswith('RB') or tag.startswith('CC') or tag.startswith('WP')
                or tag.startswith('FW') or tag.startswith('IN') or tag.startswith('DT') or tag.startswith('CD')
                or tag.startswith('RP')]
    else:  # cl == 3
        text = [word for word, tag in pos_tags if tag.startswith('JJ') or tag.startswith('NN') or tag.startswith('VB')
                or tag.startswith('RB') or tag.startswith('WP') or tag.startswith('CD')]
    return text


def feature_selection(text, cl):
    """
    Applies feature selection on the text based on the lexical properties of the words.

    Args:
    text (list of str): data of 'Phrase'.

    Returns:
    list of str: The transformed text.
    """
    selected_corpus = []
    for doc in text:
        selected_corpus.append(select_words(doc, cl))
    return selected_corpus
