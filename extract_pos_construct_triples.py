# this code retrieves tree structures from sentences to facilitate ontology engineering and locates the data objects
# (for ontology learning) see https://stanfordnlp.github.io/stanfordnlp/data_objects.html
#
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


# advanced entity extraction - extracts MWEs like "Nick Fury", yet  multiple subjects like "Nick Fury & Maria Hill"
# do not work correctly. It is able to detect a second subject even if not marked as such by Stanford Core to some
# degree; it will recognize 'Peter and MJ', which most likely will be flagged as 'subj cc conj', but not "Peter and
# his friends". It extracts 1 occurrence of 'subj','obj', 'root' (more than 1 require more sophisticated solutions)
# dependency_relation is a key element (word to grammar relationships); while 'nsubj', 'nsubj:pass', 'compound' and
# 'flat' characterize subjects and 'obj', 'obl' the object, relationship values are often indicated by the 'root' &
# sometimes the 'conj' dependency. Yet, there is no exhaustive list of possible dependency values, but alphabetical
# listing of the Universal Dependency project is expanded on https://universaldependencies.org/u/dep/index.html
def extract_entities(df, dependency_term):
    entities = []
    try:
        pos_entity = df.loc[df['dependency'].str.contains(dependency_term)]['sentence_position'].values[0]
        name_entity = df.loc[df['dependency'].str.contains(dependency_term)]['text'].values[0]
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


# increase quality of extraction by including Stanford's OpenIE (https://stanfordnlp.github.io/CoreNLP/openie.html)
# which generates triples from text, while Stanford NER supports mapping MWEs like "Maria" to schema.org/person
# returns five lists ordering triples from OpenIE according to (s,p,o) as well as the associated concepts and
# individuals generated through Named Entity Recognition if applicable
def ask_openie_ner_service(df):
    str_repr = ' '.join(df['text'])

    openie_output = inst.annotate(str_repr, properties={
        'annotators': 'tokenize,ssplit,pos,depparse,parse, openie, ner',
        'outputFormat': 'json'
    })

    subjects_openie = []
    objects_openie = []
    relations_openie = []
    token_ner = []
    metadata_ner = []

    # populate lists for triples
    try:
        for ii in range(0, len(openie_output['sentences'][0]['openie'])):
            subjects_openie.append(openie_output['sentences'][0]['openie'][ii]['subject'])
            relations_openie.append(openie_output['sentences'][0]['openie'][ii]['relation'])
            objects_openie.append(openie_output['sentences'][0]['openie'][ii]['object'])
    except Exception as e:
        logger.error(e)
        pass

    # populate lists for metadata
    try:
        for jj in range(0, len(openie_output['sentences'][0]['entitymentions'])):
            token_ner.append(openie_output['sentences'][0]['entitymentions'][jj]['text'])
            metadata_ner.append(openie_output['sentences'][0]['entitymentions'][jj]['ner'])

    except Exception as e:
        logger.error(e)
        pass

    return(subjects_openie, relations_openie, objects_openie, metadata_ner, token_ner)


# aligns the extraction results from extract_entities() with the services from OpenIE and NER
def align_extraction_results(df):

    # extract subjects per sentence
    subjects_ext = extract_entities(df, 'subj')
    if (subjects_ext == "Entity Extraction failed"):
        subjects_ext = []

    # extract the root of the sentence
    relations_ext = extract_entities(df, 'root')
    if (relations_ext == "Entity Extraction failed"):
        relations_ext = []

    # extract objects per sentence; if 'obj' is not recognized, the object is most likely a place,
    # which is indicated by 'obl', so try this instead
    objects_ext = extract_entities(df, 'obj')
    if (objects_ext == "Entity Extraction failed"):
        objects_ext = extract_entities(df, 'obl')
        if (objects_ext == "Entity Extraction failed"):
            objects_ext = []

    # the Pandas Series are used to rectify the sometimes missing index
    triples_extracted_df = pandas.DataFrame(columns=['Subject', 'Predicate', 'Object'])
    triples_extracted_df['Subject'] = pandas.Series(subjects_ext)
    triples_extracted_df['Predicate'] = pandas.Series(relations_ext)
    triples_extracted_df['Object'] = pandas.Series(objects_ext)

    # call Stanford extraction services
    service_response = ask_openie_ner_service(df)

    triples_openie_df = pandas.DataFrame(columns=['Subject', 'Predicate', 'Object'])
    triples_openie_df['Subject'] = pandas.Series(service_response[0])
    triples_openie_df['Predicate'] = pandas.Series(service_response[1])
    triples_openie_df['Object'] = pandas.Series(service_response[2])

    metadata_ner_df = pandas.DataFrame(columns=['Concept', 'Individual'])
    metadata_ner_df['Concept'] = pandas.Series(service_response[3])
    metadata_ner_df['Individual'] = pandas.Series(service_response[4])

    return(triples_extracted_df, triples_openie_df, metadata_ner_df)



stanfordnlp.download('en') # technically, this line needs to be executed only once

inst = StanfordCoreNLP('http://localhost:9000')
nlp = stanfordnlp.Pipeline()


# eventually, this code snippets generates two data frames holding valuable information to go forward; triples_df
# contains the individuals, but before creation of RDF triples can be commenced, a meta schema must be generated
# to set the namespaces for URIRefs and BNodes i.e. with the help of python's rdflib, so metadata_df holds all
# concepts mapped to individuals, so it can be processed in a lookup dictionary
# (limit to gather knowledge for 1 movie at a time, due to performance when appending data frames and manageability)
triples = []
triples_df = pandas.DataFrame(columns=['Subject', 'Predicate', 'Object'])
metadata_df = pandas.DataFrame(columns=['Concept', 'Individual'])
for i in range(22, len(plot_dict)):
    doc = nlp(plot_dict[urls[i]])
    for y in range(0, len(doc.sentences)):
        annotated_words = doc.sentences[y].words
        df = get_shallow_parsing_structures(annotated_words)
        extracted_results = align_extraction_results(df)
        triples_df = triples_df.append(extracted_results[0])
        triples_df = triples_df.append(extracted_results[1])
        metadata_df = metadata_df.append(extracted_results[2])
