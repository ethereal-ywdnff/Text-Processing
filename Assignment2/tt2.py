import spacy

nlp = spacy.load("en_core_web_sm")

def extract_features(sentence):
    doc = nlp(sentence)
    features = {
        'num_tokens': len(doc),
        'num_verbs': sum(1 for token in doc if token.pos_ == 'VERB'),
        'num_nouns': sum(1 for token in doc if token.pos_ == 'NOUN'),
        # 更多特征...
    }
    return features

sentence = "The movie was surprisingly good."
features = extract_features(sentence)
