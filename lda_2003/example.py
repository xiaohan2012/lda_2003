from __future__ import division

import numpy as np
from collections import Counter

from util import (load_line_corpus, vectorize_docs)
from lda import train

docs, vocab = vectorize_docs(load_line_corpus('data/nips-2014.dat'))

K = 4
V = len(vocab)

# initialize alpha
alpha = np.random.rand(K) * 2


# initialize beta
# add some prior on the topics
topic_words = [['bandit', 'regret'],
               ['infer', 'exponenti'],
               ['neural', 'network'],
               ['algorithm']]

vocab_inv = {w: i
             for i, w in vocab.items()}

beta1 = np.zeros((K, V), dtype=np.float64)
for k in xrange(K):
    ids = [vocab_inv[w] for w in topic_words[k]]
    if len(ids) > 0:
        beta1[k, ids] = 1. / len(ids)

words = np.concatenate(docs)

beta2 = np.zeros(V, dtype=np.float64)

counter = Counter(words)
for id_, c in counter.items():
    beta2[id_] = c + 0.1

beta2 /= (len(words) + 0.1 * len(counter))

beta2 = np.random.rand(K, V)
beta2 /= beta2.sum(axis=1)[:, None]

beta = 0.2 * beta1 + 0.8 * beta2


assert np.abs(beta.sum(axis=1) - np.ones(K)).max() < 1e-5,\
    beta.sum(axis=1)

(alpha, beta, ips, phi, lower_bound_values)\
    = train(docs, alpha, beta, K, V, max_iter=100)

for i in xrange(K):
    words = [vocab[id_] for id_ in np.argsort(beta[i])[::-1][:10]]
    print("Topic {}: {}".format(i, ' '.join(words)))
