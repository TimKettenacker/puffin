# the code retrieves tree structures from sentences to facilitate ontology engineering
# For a general introduction to computational linguistics refer to
# https://www.analyticsvidhya.com/blog/2017/12/introduction-computational-linguistics-dependency-trees/?utm_source=blog&utm_medium=stanfordnlp-nlp-library-python

# before running start the stanford core server in the respective directory, following the commentary in this manual:
# https://stackoverflow.com/questions/32879532/stanford-nlp-for-python
# cd Documents
# cd stanford-corenlp-full-2018-10-05
# java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000

import stanfordnlp
import pandas
from pycorenlp import StanfordCoreNLP

# advanced entity extraction - extracts MWEs like "Nick Fury", yet  multiple subjects like "Nick Fury and Maria Hill"
# do not work correctly. It is able to detect a second subject even if not marked as such by Stanford Core
# extracts 1 occurrence of 'subj' or 'obj', sentences with more than 1 require even more complex solutions
def extract_entities(df, subj_or_obj):
    entities = []
    try:
        pos_entity = df.loc[df['dependency'].str.contains(subj_or_obj)]['sentence_position'].values[0]
        name_entity = df.loc[df['dependency'].str.contains(subj_or_obj)]['text'].values[0]
    except:
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
    except:
        pass
    # find additional entities
    try:
        if nxt_pos_dpndcy.values[0] in ["cc"]:
            upos_check = df.loc[df.loc[:, 'sentence_position'] == str(int(pos_entity) + 2), 'upos']
            if upos_check.values[0] in ["PROPN", "PRON"]:
                second_entity = df.loc[df.loc[:, 'sentence_position'] == str(int(pos_entity) + 2), 'text']
                entities.append(second_entity.values[0])
    except:
        pass
    return entities


stanfordnlp.download('en')

inst = StanfordCoreNLP('http://localhost:9000')

nlp = stanfordnlp.Pipeline()
doc = nlp(plot_dict[urls[22]])

# data objects need to be extracted for ontologies, see https://stanfordnlp.github.io/stanfordnlp/data_objects.html
# dependency_relation is a key element (word to grammar relationships); while 'nsubj', 'nsubj:pass', 'compound' and
# 'flat' characterize subjects and 'obj', 'obl' the object, relationship values are often indicated by the 'root' and
# sometimes the 'conj' dependency. Yet, there is no exhaustive list of possible dependency values, but the alphabetical
# listing of the Universal Dependency project is expanded on https://universaldependencies.org/u/dep/index.html
# data object governor means a word that another word is dependent on

# map data objects
# try recognizing entities and map relations

annotated_words = doc.sentences[1].words
df = pandas.DataFrame()

word_index = []
word_text = []
word_upos = []
word_dependency_relation = []
word_governor = []

for u in range(0, len(annotated_words)):
    word_index.append(annotated_words[u].index)
    word_text.append(annotated_words[u].text)
    word_upos.append(annotated_words[u].upos)
    word_dependency_relation.append(annotated_words[u].dependency_relation)
    word_governor.append(annotated_words[u].governor)

df[0] = word_index
df[1] = word_text
df[2] = word_upos
df[3] = word_dependency_relation
df[4] = word_governor

# clean up data frame
df = df.loc[~df[2].str.contains('PUNCT')]
df.columns = ['sentence_position', 'text', 'upos', 'dependency', 'governor_position']

# extract subjects per sentence
subjects = extract_entities(df, 'subj')

# extract objects per sentence
objects = extract_entities(df, 'obj')

# extract the root of the sentence
root_rel = extract_entities(df, 'root')

# compare to output of openIE
str_repr = ' '.join(word_text)

openie_output = inst.annotate(str_repr, properties={
  'annotators': 'tokenize,ssplit,pos,depparse,parse,openie',
  'outputFormat': 'json'
  })

# in fact, there will be multiple sentences as Open IE gives variations of relations,
# so we need a loop to write to a list and compare the list contents with the one from above
openie_output['sentences'][0]['openie']