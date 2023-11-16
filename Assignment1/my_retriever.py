from math import sqrt, log


class Retrieve:
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.max_tf_doc = {}  # max term frequency of each word in all documents
        self.query_weights = {}
        self.doc_ids = set()
        self.num_docs = None  # number of documents
        self.doc_length = {}  # length of vector
        self.compute_documents_info()

    """compute_documents_info(self)
    This function is to compute the required parameters defined in the initialisation function."""
    def compute_documents_info(self):
        for word in self.index:
            self.doc_ids.update(self.index[word])
        self.num_docs = len(self.doc_ids)
        # compute the max term frequency of each word in all documents
        for word in self.index:
            for no in self.index[word]:
                term_freq = self.index[word][no]
                if no not in self.doc_length:
                    self.max_tf_doc[no] = 0
                if term_freq > self.max_tf_doc[no]:
                    self.max_tf_doc[no] = term_freq
        for word in self.index:
            for no in self.index[word]:
                term_freq = self.index[word][no]
                # Compute length of vector
                if no not in self.doc_length:
                    self.doc_length[no] = 0
                if self.term_weighting == 'binary':
                    self.doc_length[no] += 1
                elif self.term_weighting == 'tf':
                    # ntf = self.max_tf_normalization(0.45, term_freq, self.max_tf_doc[no])
                    # self.doc_length[no] += ntf ** 2
                    wf = 1 + log(term_freq)
                    self.doc_length[no] += wf ** 2
                elif self.term_weighting == 'tfidf':
                    df = len(self.index[word])
                    idf = log(self.num_docs / df)
                    ntf = self.max_tf_normalization(0.2, term_freq, self.max_tf_doc[no])
                    self.doc_length[no] += (ntf * idf) ** 2
        # Extraction of square root
        for no in self.doc_length:
            self.doc_length[no] = sqrt(self.doc_length[no])

    """compute_cos(self, query_weights)
    This function is for terming weight of tf and tfidf to compute the cosine similarity."""
    def compute_cos(self, query_weights):
        similarity = {i: 0 for i in self.doc_ids}
        for word, weight in query_weights.items():
            if word in self.index:
                for no, term_freq in self.index[word].items():
                    if self.term_weighting == 'tfidf':
                        df = len(self.index[word])
                        idf = log(self.num_docs / df)
                        ntf = self.max_tf_normalization(0.2, term_freq, self.max_tf_doc[no])
                        similarity[no] += weight * ntf * idf
                    else:  # If term weighting is 'tf', idf doesn't need to be computed
                        # ntf = self.max_tf_normalization(0.45, term_freq, self.max_tf_doc[no])
                        # similarity[no] += weight * ntf
                        wf = 1 + log(term_freq)
                        similarity[no] += weight * wf
        # Compute the similarity (divide by the length of vector 'doc_length')
        for sim in similarity:
            similarity[sim] /= self.doc_length[sim]

        return similarity

    """max_tf_normalization(self, a, tf, tf_max)
    This function is use maximum tf normalization to normalize tf. 
    Smoothing factor a can be modified according to the performance"""
    def max_tf_normalization(self, a, tf, tf_max):
        return a + (1 - a) * tf / tf_max

    """using_binary(self, query)
    This function is to use the binary scheme to perform the retrieval."""
    def using_binary(self, query):
        # Create a dictionary that stores words and occurrence
        for word in query:
            # In the binary weighting calculation, whenever a word is included in the query
            # (no matter how many times it occurs), its term frequency is set to 1
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
    This function is to use the tf scheme to perform the retrieval."""
    def using_tf(self, query):
        # Create a dictionary that stores words and their times of occurrence
        for word in query:
            self.query_weights[word] = self.query_weights.get(word, 0) + 1

        max_tf = max([j for i, j in self.query_weights.items()])
        for word in self.query_weights:
            if word in self.index:
                # ntf = self.max_tf_normalization(0.45, self.query_weights[word], max_tf)
                # self.query_weights[word] = ntf

                # Use wf rather than tf
                wf = 1 + log(self.query_weights[word])  # 1 + log(tf)
                self.query_weights[word] = wf
        result = self.compute_cos(self.query_weights)

        return result

    """using_tfidf(self, query)
    This function is to use the tfidf scheme to perform the retrieval."""
    def using_tfidf(self, query):
        # Create a dictionary that stores words and their times of occurrence
        for word in query:
            self.query_weights[word] = self.query_weights.get(word, 0) + 1
        max_tf = max([j for i, j in self.query_weights.items()])
        # The condition is different from that of 'tf' scheme
        # In the terming weight of 'tfidf', it needs to times the idf
        for word in self.query_weights:
            if word in self.index:
                # Use ntf rather than tf
                ntf = self.max_tf_normalization(0.2, self.query_weights[word], max_tf)
                df = len(self.index[word])
                idf = log(self.num_docs / df)
                self.query_weights[word] = ntf * idf
        result = self.compute_cos(self.query_weights)

        return result

    """for_query(self, query)
    This function uses all the functions above to return the sorted result from best similarity to worst similarity."""
    def for_query(self, query):
        result = None
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
