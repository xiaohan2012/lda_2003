import os
import codecs

import nltk
import numpy as np

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
                   for word in sent if word not in stopwords]
            docs.append(doc)

    return docs


