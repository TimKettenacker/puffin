# the code retrieves tree structures from sentences to facilitate ontology engineering
# For a general introduction to computational linguistics refer to
# https://www.analyticsvidhya.com/blog/2017/12/introduction-computational-linguistics-dependency-trees/?utm_source=blog&utm_medium=stanfordnlp-nlp-library-python

# before running start the stanford core server in the respective directory, following the commentary in this manual:
# https://stackoverflow.com/questions/32879532/stanford-nlp-for-python
# cd Documents
# cd stanford-corenlp-full-2018-10-05
# java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000
import logging
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)

import stanfordnlp
import pandas
from pycorenlp import StanfordCoreNLP

# data objects need to be extracted for ontologies, see https://stanfordnlp.github.io/stanfordnlp/data_objects.html
# dependency_relation is a key element (word to grammar relationships); while 'nsubj', 'nsubj:pass', 'compound' and
# 'flat' characterize subjects and 'obj', 'obl' the object, relationship values are often indicated by the 'root' and
# sometimes the 'conj' dependency. Yet, there is no exhaustive list of possible dependency values, but the alphabetical
# listing of the Universal Dependency project is expanded on https://universaldependencies.org/u/dep/index.html


# map data objects inside a dataframe to the output of the shallow parsing
# data object governor means a word that another word is dependent on
def get_shallow_parsing_structures(annotated_words):
    df = pandas.DataFrame()

    word_index = []
    word_text = []
    word_upos = []
    word_dependency_relation = []
    word_governor = []

    try:
        for u in range(0, len(annotated_words)):
            word_index.append(annotated_words[u].index)
            word_text.append(annotated_words[u].text)
            word_upos.append(annotated_words[u].upos)
            word_dependency_relation.append(annotated_words[u].dependency_relation)
            word_governor.append(annotated_words[u].governor)
    except Exception as e:
        logger.error(e)
        pass

    df[0] = word_index
    df[1] = word_text
    df[2] = word_upos
    df[3] = word_dependency_relation
    df[4] = word_governor

    # clean up data frame
    df = df.loc[~df[2].str.contains('PUNCT')]
    df.columns = ['sentence_position', 'text', 'upos', 'dependency', 'governor_position']

    return(df)


# advanced entity extraction - extracts MWEs like "Nick Fury", yet  multiple subjects like "Nick Fury and Maria Hill"
# do not work correctly. It is able to detect a second subject even if not marked as such by Stanford Core to some
# degree; it will recognize 'Peter and MJ', which most likely will be flagged as 'subj cc conj', but not "Peter and
# his friends". It extracts 1 occurrence of 'subj','obj', sentences with more than 1 require more sophisticated solution
def extract_entities(df, subj_or_obj):
    entities = []
    try:
        pos_entity = df.loc[df['dependency'].str.contains(subj_or_obj)]['sentence_position'].values[0]
        name_entity = df.loc[df['dependency'].str.contains(subj_or_obj)]['text'].values[0]
    except Exception as e:
        logger.error(e)
        return "Entity Extraction failed"
    # get mwes
    try:
        nxt_pos_dpndcy = df.loc[df.loc[:, 'sentence_position'] == str(int(pos_entity) + 1), 'dependency']
        if len(nxt_pos_dpndcy) == 0:
            entities.append(name_entity)
        elif nxt_pos_dpndcy.values[0] in ["fixed", "flat", "compound"]:
            mwe = df.loc[df.loc[:, 'sentence_position'] == str(int(pos_entity) + 1), 'text']
            entity = name_entity + " " + mwe.values[0]
            entities.append(entity)
        elif nxt_pos_dpndcy.values[0] not in ["fixed", "flat", "compound"]:
            entities.append(name_entity)
    except Exception as e:
        logger.error(e)
        pass
    # find additional entities
    try:
        if nxt_pos_dpndcy.values[0] in ["cc"]:
            upos_check = df.loc[df.loc[:, 'sentence_position'] == str(int(pos_entity) + 2), 'upos']
            if upos_check.values[0] in ["PROPN", "PRON"]:
                second_entity = df.loc[df.loc[:, 'sentence_position'] == str(int(pos_entity) + 2), 'text']
                entities.append(second_entity.values[0])
    except Exception as e:
        logger.error(e)
        pass
    return entities


# increase quality of extraction services by including Stanford's openie
# https://stanfordnlp.github.io/CoreNLP/openie.html
def get_openie_values(df):
    str_repr = ' '.join(df['text'])

    openie_output = inst.annotate(str_repr, properties={
        'annotators': 'tokenize,ssplit,pos,depparse,parse,openie',
        'outputFormat': 'json'
    })

    subjects_openie = []
    objects_openie = []
    relations_openie = []

    try:
        for i in range(0, len(openie_output['sentences'][0]['openie'])):
            subjects_openie.append(openie_output['sentences'][0]['openie'][i]['subject'])
            objects_openie.append(openie_output['sentences'][0]['openie'][i]['object'])
            relations_openie.append(openie_output['sentences'][0]['openie'][i]['relation'])
    except Exception as e:
        logger.error(e)
        pass

    return(subjects_openie, objects_openie, relations_openie)


# generate triples by combining both extraction methods
def combine_extraction_results_into_triples(df):
    # extract subjects per sentence
    subjects_ext = extract_entities(df, 'subj')
    if (subjects_ext == "Entity Extraction failed"):
        subjects_ext = []

    # extract objects per sentence; if 'obj' is not recognized, the object is most likely a place,
    # which is indicated by 'obl', so try this instead
    objects_ext = extract_entities(df, 'obj')
    if (objects_ext == "Entity Extraction failed"):
        objects_ext = extract_entities(df, 'obl')
        if (objects_ext == "Entity Extraction failed"):
            objects_ext = []

    # extract the root of the sentence
    relations_ext = extract_entities(df, 'root')
    if (relations_ext == "Entity Extraction failed"):
        relations_ext = []

    # call openie
    openie_out = get_openie_values(df)
    subjects_openie = openie_out[0]
    objects_openie = openie_out[1]
    relations_openie = openie_out[2]
    subjects = list(set(subjects_ext + subjects_openie))
    objects = list(set(objects_ext + objects_openie))
    relations = list(set(relations_ext + relations_openie))

    # create cartesian product
    triple_list = [subjects, relations, objects]
    cart_prod = [(s, p, o) for s in triple_list[0] for p in triple_list[1] for o in triple_list[2]]

    return(cart_prod)



stanfordnlp.download('en') # technically, this line needs to be executed only once

inst = StanfordCoreNLP('http://localhost:9000')
nlp = stanfordnlp.Pipeline()

# for the time being, generate knowledge about 1 movie only
triples = []
for i in range(22, len(plot_dict)):
    doc = nlp(plot_dict[urls[i]])
    for y in range(0, len(doc.sentences)):
        annotated_words = doc.sentences[y].words
        df = get_shallow_parsing_structures(annotated_words)
        triples.append(combine_extraction_results_into_triples(df))

# variable "triples" contains the instance data, before creation of RDF triples can be commenced, a meta schema
# must be generated to set the namespaces for URIRefs and BNodes with the help of python's rdflib or similar,
# in order to map values like "Maria Hill" to schema.org/person etc. - could a full-stack application be useful
# to identify the important content (selecting the relevant triples)?
# maybe LDA and word2vec to get a sense of (neighboring) topics? -> Then checking available ontologies manually
# instance data can be mapped to owl after ontology is created, "triples" variable could be added as individuals in
# https://owlready2.readthedocs.io/en/latest/ - can read the RDF triples and other ontologies / vocabulary,
# is able to alter and manage them in an integrated quad store
# http://jens-lehmann.org/files/2014/perspectives_on_ontology_learning.pdf