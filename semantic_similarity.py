# the code snippets presented here are most useful when it comes to applying queries on text; map keywords input
# on apparent semantic relatedness of the documents (in how far can it answer questions that are implicit in text)
# https://radimrehurek.com/gensim/auto_examples/core/run_similarity_queries.html#sphx-glr-auto-examples-core-run-similarity-queries-py
# before running start the stanford core server in the respective directory, following the commentary in this manual:

# https://stackoverflow.com/questions/32879532/stanford-nlp-for-python
# cd Documents
# cd stanford-corenlp-full-2018-10-05
# java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000

#import sys
import os
import logging
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)
import owlready2
from owlready2 import *
from collections import defaultdict
from pycorenlp import StanfordCoreNLP
from fuzzywuzzy import fuzz

# load ontology
onto_path.append(str(os.getcwd() + '/data'))
onto = get_ontology("ontology_Homecoming.owx").load()

# question = sys.argv[1]
questions1 = ["Who killed Quentin Beck?", "How does Quentin Beck die?", "Quentin Beck death"]
questions2 = ["Where is the school travelling to?", "Where is the school going?", "which locations are visited?"]
# 1. create a list of adverbs describing the intent: WHERE -> location, WHO -> person
# fire all you have got to get a grasp of what is asked for, like NER, openie, word embedding etc.
# then load all docs from the wikipedia page and combine with ontology
# return ontology on top prio if it is exactly matching
# check wikidata for word families, i.e. "death" also known as ... and check
# create hypernyms (https://stackoverflow.com/questions/19258652/how-to-get-synonyms-from-nltk-wordnet-python)

# 1. intents: get det and advmod and their governors
# check if they can be linked by WHERE -> location or who -> governor person or organization
# with additional information about the governor, i.e. a lookup in the ontology
# https://universaldependencies.org/u/pos/index.html
def extract_question_indications(list_of_annotations, list_of_tokens):
    findings = {'subj_of_intent' : '',
                'root_of_intent': '',
                'obj_of_intent' : '',
                'concept_of_target' : ''}

    for annotated_word in list_of_annotations:
        if 'subj' in annotated_word['dep']:
            findings['subj_of_intent'] = annotated_word['dependentGloss']
        if 'ROOT' in annotated_word['dep']:
            findings['root_of_intent'] = annotated_word['dependentGloss']
        if 'obj' in annotated_word['dep']:
            findings['obj_of_intent'] = annotated_word['dependentGloss']

    for tokenized in list_of_tokens:
        if findings['subj_of_intent'] == tokenized['originalText']:
            findings['subj_of_intent'] = tokenized['lemma']
        if findings['root_of_intent'] == tokenized['originalText']:
            findings['root_of_intent'] = tokenized['lemma']
        if findings['obj_of_intent'] == tokenized['originalText']:
            findings['obj_of_intent'] = tokenized['lemma']

    for value in findings.values():
        if value in words_indicating_concepts.keys():
            findings['concept_of_target'] = words_indicating_concepts[value]

    return(findings)


def search_ontology_for_involved_entities(findings, startPointIsObject):

    spo_dict = defaultdict(list)

    involved_subjects = []
    involved_predicates = []
    involved_objects = []

    if startPointIsObject == True:
        label = '*' + findings['obj_of_intent'] + '*'
        ontology_item = onto.search_one(label=label)
        for s, p in ontology_item.get_inverse_properties():
            involved_subjects.append(s.label.first())
            involved_predicates.append(p.label.first())

    if startPointIsObject == False:
        label = '*' + findings['subj_of_intent'] + '*'
        ontology_item = onto.search_one(label=label)
        for p, o in ontology_item.get_properties():
            involved_predicates.append(p.label.first())
            involved_objects.append(o.label.first())

    spo_dict['involved_subjects'] = involved_subjects
    spo_dict['involved_predicates'] = involved_predicates
    spo_dict['involved_objects'] = involved_objects

    return(spo_dict)


words_indicating_concepts = {'where':'location',
                             'who':'person',
                             'whom':'person',
                             'when':'event',
                             'what':'thing'}


inst = StanfordCoreNLP('http://localhost:9000')
nlp_output = inst.annotate(questions1[0], properties={
        'annotators': 'tokenize,ssplit,pos,depparse,parse, openie, ner',
        'outputFormat': 'json'
    })
findings = extract_question_indications(nlp_output['sentences'][0]['enhancedPlusPlusDependencies'],
                                   nlp_output['sentences'][0]['tokens'])
found_entities = {}
for i in range(0, len(nlp_output['sentences'][0]['entitymentions'])):
    found_entities.update({nlp_output['sentences'][0]['entitymentions'][i]['ner']:
                     nlp_output['sentences'][0]['entitymentions'][i]['text']})

# check whether the entities found are matching the subject or object of the question;
# this information will be carried along to ontology lookup to steer the direction of the query,
# in particular, getting all properties or all inverse properties
startPointIsObject = False
for foundling in found_entities.values():
    if(fuzz.partial_ratio(findings['obj_of_intent'], found_entities.values())> 50):
        findings['obj_of_intent'] = foundling
        startPointIsObject = True
    if((fuzz.partial_ratio(findings['subj_of_intent'], found_entities.values())> 50)):
        findings['subj_of_intent'] = foundling

spo_dict = search_ontology_for_involved_entities(findings, startPointIsObject)
# next, it is possible to do a fuzzy check to see whether i.e. the root is in the involved predicates
# also, check if the individuals are instances_of concept_of_target

# make this an else statement (owl.ObjectProperty is recognized as a special variable and works for checking
# if it is present, as opposed to schema.org.person etc.)
t = onto.search_one(label = '*kills*')
class_or_property = t.is_a
if (owl.ObjectProperty in class_or_property) == True:
    for x, y in t.get_relations():
        print(x.label, t.label, y.label)
        #['Quentin Beck'], ['kills'], ['Fire Elemental']
        #['drones'], ['kills'], ['Quentin Beck']
        #[...]

for x in onto.individuals():
    print(x.label, x.is_instance_of)
    #['storm']['schema.org.object']
    #['Maria Hill']['schema.org.person']
    #[...]


# the different wdts look for a type of relationship and could be used to narrow down the context
# in follow-ups, similar to https://stackoverflow.com/questions/55981277/how-to-filter-wikidata-labels-in-concept-search
#
# SELECT DISTINCT ?item {
#     VALUES ?searchTerm { "death" }
#     SERVICE wikibase:mwapi {
#         bd:serviceParam wikibase:api "EntitySearch".
#         bd:serviceParam wikibase:endpoint "www.wikidata.org".
#         bd:serviceParam wikibase:limit 3 .
#         bd:serviceParam mwapi:search ?searchTerm.
#         bd:serviceParam mwapi:language "en".
#         ?item wikibase:apiOutputItem mwapi:item.
#         ?num wikibase:apiOrdinal true.
#     }
#     ?item (wdt:P279|wdt:P31|wdt:P361|wdt:P828|wdt:P910|wdt:1659) ?type
# }
# ORDER BY ?searchTerm ?num

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
