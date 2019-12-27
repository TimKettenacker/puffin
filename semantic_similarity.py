# the code snippets presented here are most useful when it comes to applying queries on text; map keywords input
# on apparent semantic relatedness of the documents (in how far can it answer questions that are implicit in text)
# https://radimrehurek.com/gensim/auto_examples/core/run_similarity_queries.html#sphx-glr-auto-examples-core-run-similarity-queries-py

import gensim
from gensim import corpora
from gensim import models

text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

# remark the stop word list of gensim eliminated too much words that were actually useful
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist] for document in text_corpus]
# remark: as opposed to the standard of throwing out words that only occur once or twice, I'm keeping all of them
# so the code below #--- is not needed
#--- Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

#--- Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint.pprint(processed_corpus)

# associate each word in corpus with unique id
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)
pprint.pprint(dictionary.token2id)
# a new document is tested against the bag of word model dictionary above
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)
print('the first tuple says id 0 is present one time in new_doc and id 1 is 1 time present, the last one isnt there')
# convert everything into that logic
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
pprint.pprint(bow_corpus)

# now, a model can be applied - here its tf-idf powered by bag of words
tfidf = models.TfidfModel(bow_corpus)
print(tfidf[[(0, 1), (1, 1)]])
# latent semantic indexing (coining the terms to its concepts) should be used in combination with tfs idf
# https://stats.stackexchange.com/questions/99132/alternatives-to-bag-of-words-based-classifiers-for-text-classification
# by setting num_topics to 2, only 2 topics are being created, each being a mixture of the words
corpus_tfidf = tfidf[bow_corpus]
for doc in corpus_tfidf:
    print(doc)
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)  # initialize an LSI transformation
corpus_lsi = lsi_model[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
lsi_model.print_topics()
for doc, as_text in zip(corpus_lsi, text_corpus):
    print(doc, as_text)


#
flat_list = [item for sublist in triples for item in sublist]
text_corpus = [' '.join(i) for i in flat_list]
texts = [[word for word in document.lower().split()] for document in text_corpus]
dictionary = corpora.Dictionary(texts)
bow_corpus = [dictionary.doc2bow(text) for text in texts]
