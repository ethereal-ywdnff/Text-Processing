from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 你的字典数据
dictionary = [
    ('preliminari', {1: 1, 254: 1, 825: 1, 894: 1, 1205: 1, 1235: 1}),  # 字典项1
    ('report', {1: 1, 65: 1, 146: 1, 147: 1, 196: 1, 254: 1}),  # 字典项2
    # 其他字典项
]

# 你的查询列表
queries = {
    1: ['articl', 'exist', 'deal', 'tss', 'time', 'share', 'system'],  # 查询1
    2: ['interest', 'articl', 'written', 'priev', 'udo', 'pooch'],   # 查询2
    # 其他查询
}

# 构建TF-IDF模型
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(queries[q]) for q in queries])

# 计算余弦相似度
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_vectorizer.transform([" ".join(doc[1].keys()) for doc in dictionary]))

# 找到每个查询的最相似的10个文档
top_similar_documents = {}
for query_id, similarity_scores in enumerate(cosine_similarities):
    top_document_indices = np.argsort(similarity_scores)[::-1][:2]
    top_similar_documents[query_id] = [(dictionary[i][0], similarity_scores[i]) for i in top_document_indices]

# 打印结果
for query_id, similar_documents in top_similar_documents.items():
    print(f"Query {query_id}:")
    for document, similarity_score in similar_documents:
        print(f"  Document: {document}, Similarity: {similarity_score:.4f}")
