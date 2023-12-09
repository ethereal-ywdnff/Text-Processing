import math


class Retrieve:

    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.doc_ids = set()
        self.doc_lengths = {}
        self.compute_documents_info()

    def compute_documents_info(self):
        for term in self.index:
            for doc_id in self.index[term]:
                self.doc_ids.add(doc_id)
                freq = self.index[term][doc_id]
                if doc_id not in self.doc_lengths:
                    self.doc_lengths[doc_id] = 0
                if self.term_weighting == 'binary':
                    self.doc_lengths[doc_id] += 1
                elif self.term_weighting == 'tf':
                    self.doc_lengths[doc_id] += freq
                elif self.term_weighting == 'tfidf':
                    doc_freq = len(self.index[term])
                    idf = math.log(len(self.doc_ids) / doc_freq)
                    self.doc_lengths[doc_id] += (freq * idf) ** 2
        for doc_id in self.doc_lengths:
            self.doc_lengths[doc_id] = math.sqrt(self.doc_lengths[doc_id])

    def compute_query_weights(self, query):
        query_weights = {}
        for term in query:
            query_weights[term] = query_weights.get(term, 0) + 1
        if self.term_weighting == 'binary':
            for term in query_weights:
                query_weights[term] = 1
        elif self.term_weighting == 'tfidf':
            num_docs = len(self.doc_ids)
            for term in query_weights:
                if term in self.index:
                    doc_freq = len(self.index[term])
                    query_weights[term] = (1 + math.log(query_weights[term])) * math.log(num_docs / doc_freq)
        return query_weights

    def compute_term_weight(self, term, doc_id, freq):
        if self.term_weighting == 'binary':
            return 1
        elif self.term_weighting == 'tf':
            return freq
        elif self.term_weighting == 'tfidf':
            num_docs = len(self.doc_ids)
            doc_freq = len(self.index[term])
            idf = math.log(num_docs / doc_freq)
            return freq * idf

    # def using_tf(self, query):
    #     query_weights = {}
    #     for term in query:
    #         query_weights[term] = query_weights.get(term, 0) + 1
    #     scores = {doc_id: 0 for doc_id in self.doc_ids}
    #     for term, weight in query_weights.items():
    #         if term in self.index:
    #             for doc_id, freq in self.index[term].items():
    #                 scores[doc_id] += weight * freq
    #     for doc_id in scores:
    #         scores[doc_id] /= self.doc_lengths[doc_id]
    #     return scores

    def for_query(self, query):
        query_weights = self.compute_query_weights(query)
        scores = {doc_id: 0 for doc_id in self.doc_ids}
        for term, weight in query_weights.items():
            if term in self.index:
                for doc_id, freq in self.index[term].items():
                    scores[doc_id] += weight * self.compute_term_weight(term, doc_id, freq)
        for doc_id in scores:
            scores[doc_id] /= self.doc_lengths[doc_id]
        # scores = self.using_tf(query)
        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, _ in ranked_docs]
