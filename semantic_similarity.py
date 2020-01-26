# the code snippets presented here are most useful when it comes to applying queries on text; map keywords input
# on apparent semantic relatedness of the documents (in how far can it answer questions that are implicit in text)
# https://radimrehurek.com/gensim/auto_examples/core/run_similarity_queries.html#sphx-glr-auto-examples-core-run-similarity-queries-py
# before running start the stanford core server in the respective directory, following the commentary in this manual:

# https://stackoverflow.com/questions/32879532/stanford-nlp-for-python
# cd Documents
# cd stanford-corenlp-full-2018-10-05
# java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000

import sys
import os
import logging
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)
from owlready2 import *
from collections import defaultdict
from pycorenlp import StanfordCoreNLP
from fuzzywuzzy import fuzz
from itertools import chain
from nltk.corpus import wordnet
from bs4 import BeautifulSoup
import requests
import re
from gensim import corpora
from gensim import models
from gensim import similarities

# load ontology
onto_path.append(str(os.getcwd() + '/data'))
onto = get_ontology("ontology_Homecoming.owx").load()


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

    triples = []
    involved_subjects = []
    involved_predicates = []
    involved_objects = []


    if startPointIsObject == True:
        label = '*' + findings['obj_of_intent'] + '*'
        ontology_item = onto.search_one(label=label)
        for s, p in ontology_item.get_inverse_properties():
            involved_subjects.append(s.label.first())
            involved_predicates.append(p.label.first())

    # if it is not clear what the object of intent is, it is a bit fiddly to base the search on;
    # then, all outgoing properties and pointers to objects are returned:
    if startPointIsObject == False:
        label = '*' + findings['subj_of_intent'] + '*'
        ontology_item = onto.search_one(label=label)
        for p in ontology_item.get_properties():
            try:
                for s, o in p.get_relations():
                    if s.label.first() in ontology_item.label:
                        triples.append(s.label.first() +
                                  '%' + p.label.first() +
                                  '%' + o.label.first())
            except:
                pass

        triples = list(map(lambda x: x.split('%'), triples))
        involved_subjects = []
        involved_objects = []
        for triple in triples:
            involved_subjects.append(triple[0])
            involved_predicates.append(triple[1])
            involved_objects.append(triple[2])

    spo_dict['involved_subjects'] = involved_subjects
    spo_dict['involved_predicates'] = involved_predicates
    spo_dict['involved_objects'] = involved_objects

    return(spo_dict)


def fuzzy_match_on_target(spo_dict):
    indv = ''
    intended_concept_confirmed = False
    for predicate in enumerate(spo_dict['involved_predicates']):
        if fuzz.ratio(predicate[1], findings['root_of_intent']) > 60:
            if startPointIsObject == True:
                indv = spo_dict['involved_subjects'][predicate[0]]
            if startPointIsObject == False:
                indv = spo_dict['involved_objects'][predicate[0]]
            if (fuzz.ratio(findings['concept_of_target'], str(indv_concept_mapping[indv])) > 50):
                intended_concept_confirmed = True
    return (indv, intended_concept_confirmed)


def search_ontology_for_property(word, findings, startPointIsObject):
    spo_dict = defaultdict(list)

    label = '*' + word + '*'
    result = onto.search_one(label=label)
    if result:
        if (owl.ObjectProperty in result.is_a) == True:
            spo_dict = search_ontology_for_involved_entities(findings, startPointIsObject)
    else:
        spo_dict = None
    return(spo_dict)


# reclycled from get_plot_outline.py
def retrieve_plot(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")

    # extract the sequence of sections
    # this is needed to stitch the regex together to exactly locate the plot description
    # starting of Plot: <span class="mw-headline" id="Plot">Plot</span> ... </h2>\n<p>
    # closing of Plot: </p>\n<h2><span class="mw-headline" id="Cast">Cast</span>
    sections = [tag.get_text() for tag in soup.find_all("span", class_="mw-headline")]
    list_index_plot = sections.index("Plot")

    content = soup.getText().splitlines()
    regex_res = re.findall('Plot(.+?)' + sections[list_index_plot + 1] + '', str(content))
    plot = regex_res.pop(1)
    return(plot)


words_indicating_concepts = {'where':'location',
                             'who':'person',
                             'whom':'person',
                             'when':'event',
                             'what':'thing'}

question = sys.argv[1]

inst = StanfordCoreNLP('http://localhost:9000')
nlp_output = inst.annotate(question, properties={
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
# in particular, getting all properties or all inverse properties in the ontology
startPointIsObject = False
for foundling in found_entities.values():
    if(fuzz.partial_ratio(findings['obj_of_intent'], found_entities.values())> 50):
        findings['obj_of_intent'] = foundling
        startPointIsObject = True
    if((fuzz.partial_ratio(findings['subj_of_intent'], found_entities.values())> 50)):
        findings['subj_of_intent'] = foundling

indv_concept_mapping = defaultdict()
for x in onto.individuals():
    indv_concept_mapping[x.label.first()] = x.is_instance_of.first()
spo_dict = search_ontology_for_involved_entities(findings, startPointIsObject)


# do a fuzzy check to see whether the root is in the involved predicates
# if so, check if the individual the root is pointing to is an instance of concept_of_target
# (as of now, nothing happens if it is not, but one could get back to the user with the intermediate
# result and ask for closer details)
indv = fuzzy_match_on_target(spo_dict)[0]

# if there is a perceived match, return it to the console and close the program
# if none of the predicates match, try finding semantically related words in the ontology
if indv:
    print('I found the following answer to your question: ' + indv)
    sys.exit()

related_words = wordnet.synsets(findings['root_of_intent'])
related_words = set(chain.from_iterable([word.lemma_names() for word in related_words]))
for word in related_words:
    match = search_ontology_for_property(word, findings, startPointIsObject)
    if match:
        indv = fuzzy_match_on_target(spo_dict)[0]
        if indv:
            print('I found the following answer to your question: ' + indv)
            sys.exit()

# if there was no match in the ontology, default back to returning similar sentences from the plot
# so far, the ontology only covers "Spider-Man: Far from home", thus it is hardcoded
# https://radimrehurek.com/gensim/auto_examples/core/run_similarity_queries.html#sphx-glr-auto-examples-core-run-similarity-queries-py
# this code could be further include n-grams to capture 'Nick Fury', 'Quentin Beck' etc.
documents = retrieve_plot('https://en.wikipedia.org/wiki/Spider-Man:_Far_From_Home')
documents = documents.lower().split('.')
stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=round(len(texts) / 2))
vec_bow = dictionary.doc2bow(question.lower().split())
vec_lsi = lsi[vec_bow]
index = similarities.MatrixSimilarity(lsi[corpus])
sims = index[vec_lsi]
sims = sorted(enumerate(sims), key=lambda item: -item[1])
documents[sims[0][0]]

# an unexplored feature of this code lies in the enrichment through community curated content
# the different wdts look for a type of relationship and could be used to narrow down the context
# https://stackoverflow.com/questions/55981277/how-to-filter-wikidata-labels-in-concept-search
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