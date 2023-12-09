from math import log


class Retrieve:

    # Create new Retrieve object storing index and term weighting 
    # scheme. ​(You can extend this method, as required.)
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.n = 0
        self.b = []
        print(self.num_docs)

    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        # print(f"number of index : {len(self.doc_ids)}")
        return len(self.doc_ids)

    # Method performing retrieval for a single query (which is
    # represented as a list of preprocessed terms). ​Returns list
    # of doc ids for relevant docs (in rank order).
    # python IR_engine.py -s -p -o test.txt
    def for_query(self, query):
        if self.term_weighting == "binary":
            ranked_list = self.using_binary(query)
            return self.count_number(ranked_list)
        if self.term_weighting == "tfidf":
            self.use_tfidf(query)
        return list(range(2, 12))

    def using_binary(self, query):
        self.b = []
        a = []
        for word in query:
            for key, value in list(self.index.items()):
                if key == word:
                    for i, j in list(value.items()):
                        # Make sure that the number of occurrences of each word
                        # in the `query` matches that of the word in the document
                        if query.count(word) == j:
                            a.append(i)
                    self.b.append(a)
                    a = []
        return self.b

    def use_tfidf(self, query):
        vocabulary = set(word for doc in self.index for word in doc)

        # 计算文档频率
        document_frequency = {}
        for word in vocabulary:
            document_frequency[word] = sum(1 for doc in self.index if word in doc)

        # 计算逆文档频率
        total_documents = len(self.index)
        inverse_document_frequency = {word: log(total_documents / (df + 1)) for word, df in document_frequency.items()}

        # 计算TF-IDF值
        tfidf = {}
        for doc_id, doc in enumerate(self.index):
            tfidf[doc_id] = {}
            for word in doc:
                tf = doc.count(word)
                tfidf[doc_id][word] = tf * inverse_document_frequency[word]

        # 构建倒排索引
        inverted_index = {}
        for doc_id, doc in enumerate(self.index):
            for word in doc:
                if word not in inverted_index:
                    inverted_index[word] = []
                inverted_index[word].append(doc_id)

        # 查询文档
        query_document = query  # 从你的查询中获取
        query_tfidf = {}  # 计算查询文档的TF-IDF值
        for word in query_document:
            tf = query_document.count(word)
            if word in inverse_document_frequency:
                query_tfidf[word] = tf * inverse_document_frequency[word]

        # 计算查询文档与每个文档的相似性分数
        similarity_scores = {}
        for doc_id, doc_tfidf in tfidf.items():
            score = 0
            for word, tfidf_value in query_tfidf.items():
                if word in doc_tfidf:
                    score += tfidf_value * doc_tfidf[word]
            similarity_scores[doc_id] = score

        # 获取相似性分数最高的文档
        sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        top_10_documents = [doc_id for doc_id, _ in sorted_scores[:10]]

        # 输出查询结果
        # for doc_id in top_10_documents:
        return top_10_documents


    # count the occurrences of each value in all the lists
    # e.g. [[1, 2, 3], [2, 5], [3, 2, 4]] --> [2, 3, 1, 5, 4]
    def count_number(self, count_list):
        count_dict = {}
        sort = []

        # Go through all the sublists and count the number of occurrences
        for sublist in count_list:
            for number in sublist:
                count_dict[number] = count_dict.get(number, 0) + 1

        # Sort numbers according to how many times they appear
        sorted_numbers = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

        # Output the sorted numbers according to their top ten occurrences
        for number, count in sorted_numbers[:10]:
            sort.append(number)
        return sort

