from math import sqrt, log


class Retrieve:
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.query_weights = {}
        self.doc_ids = set()
        self.num_docs = 0
        self.doc_length = {}
        self.compute_documents_info()

    def compute_documents_info(self):
        for term in self.index:
            self.doc_ids.update(self.index[term])
            for num in self.index[term]:
                term_freq = self.index[term][num]
                # Compute length of vector
                if num not in self.doc_length:
                    self.doc_length[num] = 0
                if self.term_weighting == 'binary':
                    self.doc_length[num] += 1
                elif self.term_weighting == 'tf':
                    self.doc_length[num] += term_freq
                elif self.term_weighting == 'tfidf':
                    doc_freq = len(self.index[term])
                    idf = log(len(self.doc_ids) / doc_freq)
                    self.doc_length[num] += (term_freq * idf) ** 2
        self.num_docs = len(self.doc_ids)
        # Extraction of square root
        for num in self.doc_length:
            self.doc_length[num] = sqrt(self.doc_length[num])


    def using_binary(self, query):
        # Create a dictionary that stores words and occurrence
        for word in query:
            # In the binary weighting calculation, whenever a word is included in the query
            # (no matter how many times it occurs), its weight is set to 1
            self.query_weights[word] = 1
        print(self.query_weights)
        scores = {i: 0 for i in self.doc_ids}
        for word, weight in self.query_weights.items():
            if word in self.index:  # Whether the term is present in document
                for doc_id, freq in self.index[word].items():
                    scores[doc_id] += weight
        for score in scores:
            scores[score] /= self.doc_length[score]
        return scores


    def for_query(self, query):
        if self.term_weighting == 'binary':
            scores = self.using_binary(query)

        self.query_weights = {}  # make it to empty so that it can't impact the following queries
        ranked_result = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print([i for i, _ in ranked_result])
        return [i for i, _ in ranked_result]
