import nltk

# print(nltk.data.path)

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')


def extract_features(text):
    # 分词
    words = word_tokenize(text)
    print(words)

    # 移除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # 词性标注
    pos_tags = nltk.pos_tag(words)

    # 将词性标注结果转换为一句话
    tagged_sentence = ' '.join([word for word, tag in pos_tags])

    # 结合特征
    features = {
        tagged_sentence
    }
    return features

# 示例
text = "This is an amazing product but I dont like its color"
features = extract_features(text)
print(features)

