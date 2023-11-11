from math import sqrt, log


class Retrieve:
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.query_weights = {}
        self.doc_ids = set()
        self.num_docs = None  # number of documents
        self.doc_length = {}  # length of vector
        self.compute_documents_info()

    """compute_documents_info(self)
    This function is to compute the required parameters defined in the initialisation function"""
    def compute_documents_info(self):
        for word in self.index:
            self.doc_ids.update(self.index[word])
            for no in self.index[word]:
                term_freq = self.index[word][no]
                # Compute length of vector
                if no not in self.doc_length:
                    self.doc_length[no] = 0
                if self.term_weighting == 'binary':
                    self.doc_length[no] += 1
                elif self.term_weighting == 'tf':
                    self.doc_length[no] += term_freq
                elif self.term_weighting == 'tfidf':
                    df = len(self.index[word])
                    idf = log(len(self.doc_ids) / df)
                    self.doc_length[no] += (term_freq * idf) ** 2
        self.num_docs = len(self.doc_ids)
        # Extraction of square root
        for no in self.doc_length:
            self.doc_length[no] = sqrt(self.doc_length[no])

    """compute_cos(self, query_weights)
     This function is for terming weight of tf and tfidf to compute the cosine similarity"""
    def compute_cos(self, query_weights):
        similarity = {i: 0 for i in self.doc_ids}
        for word, weight in query_weights.items():
            if word in self.index:
                for no, term_freq in self.index[word].items():
                    if self.term_weighting == 'tfidf':
                        df = len(self.index[word])
                        idf = log(self.num_docs / df)
                        similarity[no] += weight * term_freq * idf
                    else:  # If term weighting is 'tf', idf doesn't need to be computed
                        similarity[no] += weight * term_freq
        # Compute the similarity (divide by the length of vector 'doc_length')
        for sim in similarity:
            similarity[sim] /= self.doc_length[sim]

        return similarity

    """using_binary(self, query)
    This function is to use the binary scheme to perform the retrieval"""
    def using_binary(self, query):
        # Create a dictionary that stores words and occurrence
        for word in query:
            # In the binary weighting calculation, whenever a word is included in the query
            # (no matter how many times it occurs), its weight is set to 1
            self.query_weights[word] = 1
        result = {i: 0 for i in self.doc_ids}
        for word, weight in self.query_weights.items():
            if word in self.index:  # Whether the term is present in the document
                for doc_id, freq in self.index[word].items():
                    result[doc_id] += 1  # Plus 1 if the word does in the document
        for res in result:
            result[res] /= self.doc_length[res]

        return result

    """using_tf(self, query)
    This function is to use the tf scheme to perform the retrieval"""
    def using_tf(self, query):
        # Create a dictionary that stores words and their times of occurrence
        for word in query:
            self.query_weights[word] = self.query_weights.get(word, 0) + 1
        result = self.compute_cos(self.query_weights)

        return result

    """using_tfidf(self, query)
        This function is to use the tfidf scheme to perform the retrieval"""
    def using_tfidf(self, query):
        # Create a dictionary that stores words and their times of occurrence
        for word in query:
            self.query_weights[word] = self.query_weights.get(word, 0) + 1
        # The condition is different from that of 'tf' scheme
        # In the terming weight of 'tfidf', it needs to times the idf
        for word in self.query_weights:
            if word in self.index:
                df = len(self.index[word])
                # Using log of tf as a term weight rather than tf
                wf = 1 + log(self.query_weights[word])  # wf = 1 + log(tf)
                idf = log(self.num_docs / df)
                self.query_weights[word] = wf * idf
        result = self.compute_cos(self.query_weights)

        return result

    """for_query(self, query)
    This function uses all the functions above to return the sorted result from best ranking to worst ranking"""
    def for_query(self, query):
        if self.term_weighting == 'binary':
            result = self.using_binary(query)

        if self.term_weighting == 'tf':
            result = self.using_tf(query)

        if self.term_weighting == 'tfidf':
            result = self.using_tfidf(query)

        self.query_weights = {}  # make it to empty so that it can't impact the following queries
        sorted_result = sorted(result.items(),
                               key=lambda sim: sim[1], reverse=True)  # Sort the result based on similarity

        return [i for i, j in sorted_result]
