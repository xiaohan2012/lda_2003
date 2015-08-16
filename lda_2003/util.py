import os
import codecs

import nltk
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

CURDIR = os.path.dirname(os.path.realpath(__file__))


def close_enough(old_x, new_x, abstol=10e-5):
    """
    Check if differentce between old_x and new_x is close enough.

    Being close enough is the same as
    the maximum difference betwen old_x and new_x is below abstol

    Parameter:
    ---------------

    old_x: numpy.ndarray
    
    new_x: numpy.ndarray
        same shape as new_x

    Return:
    -----------------
    bool

    """
    return np.abs((old_x - new_x)).max() <= abstol
    

def load_items_by_line(path):
    with codecs.open(path, 'r', 'utf8') as f:
        items = set([l.strip()
                     for l in f])
    return items

    
def load_line_corpus(path):
    docs = []
    
    stopwords = load_items_by_line(CURDIR + '/data/lemur-stopwords.txt')

    stemmer = nltk.PorterStemmer()
    
    with codecs.open(path, "r", "utf8") as f:
        for l in f:
            sents = nltk.sent_tokenize(l.strip().lower())
            tokenized_sents = map(nltk.word_tokenize, sents)
            doc = [stemmer.stem(word.lower())
                   for sent in tokenized_sents
                   for word in sent if word not in stopwords and len(word) > 2]
            docs.append(doc)

    return docs


def vectorize_docs(docs):
    """
    Parameter:
    --------
    docs: list of list of str
    
    Return:
    ------
    matrix: numpy.ndarray
    vocab: dict(id -> str)
    """
    word_set = set()
    for doc in docs:
        for w in doc:
            word_set.add(w)

    vocab = {i: w
             for i, w in enumerate(word_set)}
    vocab_inv = {w: i
                 for i, w in enumerate(word_set)}

    data = []
    for doc in docs:
        data.append(np.asarray([vocab_inv[w]
                                for w in doc],
                               dtype=np.int64))
    
    return (np.array(data, dtype=np.object), vocab)
    
    
def doc2term_matrix(docs):
    """
    Parameter:
    --------
    docs: list of list of str
    
    Return:
    ------
    numpy.ndarray
    """
    vect = CountVectorizer(min_df=1)
    return vect.fit_transform([' '.join(doc)
                               for doc in docs])
    

def convert_to_lda_c_format(docs, output_path):
    """
    docs: list of list of str
    
    """
    with codecs.open(output_path, 'w', 'utf8') as f:
        for doc in docs:
            M = len(set(doc))
            term_count_str = [u"{}:{}".format(t, c)
                              for t, c in Counter(doc).items()]
            f.write(u"{} {}\n".format(M, ' '.join(term_count_str)))
            

if __name__ == "__main__":
    docs = load_line_corpus('data/nips-2014.dat')
    convert_to_lda_c_format(docs, 'data/nips-2014-lda-c.dat')
