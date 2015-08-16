from __future__ import division

import numpy as np
from collections import Counter

from util import (load_line_corpus, vectorize_docs)
from lda import train

docs, vocab = vectorize_docs(load_line_corpus('data/nips-2014.dat'))

K = 4
V = len(vocab)

# initialize alpha
alpha = np.ones(K)
alpha /= alpha.size


################
# Initialize beta
# Seeded method proposed by the paper
################

beta = np.zeros((K, V))

n_sample_doc = 2
doc_ids = np.random.permutation(docs.size)[: n_sample_doc * K]
for i in xrange(K):
    ids = doc_ids[n_sample_doc * i: n_sample_doc * (i+1)]
    words = np.concatenate(docs[ids])
    counter = Counter(words)
    for id_, c in counter.items():
        beta[i, id_] = c + 0.1
    beta[i] += (np.random.rand(V) * 0.2)
    beta[i] /= beta[i].sum()

assert np.abs(beta.sum(axis=1) - np.ones(K)).max() < 1e-5,\
    beta.sum(axis=1)

(alpha, beta, ips, phi, lower_bound_values)\
    = train(docs, alpha, beta, K, V, max_iter=500)

for i in xrange(K):
    words = [vocab[id_] for id_ in np.argsort(beta[i])[::-1][:10]]
    print("Topic {}: {}".format(i, ' '.join(words)))
