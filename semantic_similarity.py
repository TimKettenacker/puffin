# variable "triples" contains the instance data, before creation of RDF triples can be commenced, a meta schema
# must be generated to set the namespaces for URIRefs and BNodes with the help of python's rdflib or similar,
# in order to map values like "Maria Hill" to schema.org/person etc. - could a full-stack application be useful
# to identify the important content (selecting the relevant triples)?
# maybe LDA and word2vec to get a sense of (neighboring) topics? -> Then checking available ontologies manually
# instance data can be mapped to owl after ontology is created, "triples" variable could be added as individuals in
# https://owlready2.readthedocs.io/en/latest/ - can read the RDF triples and other ontologies / vocabulary,
# is able to alter and manage them in an integrated quad store
# http://jens-lehmann.org/files/2014/perspectives_on_ontology_learning.pdf

import gensim
from gensim import corpora

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

# instead of using bag of words, latent semantic indexing should be used (coining the terms to its concepts)
# https://stats.stackexchange.com/questions/99132/alternatives-to-bag-of-words-based-classifiers-for-text-classification