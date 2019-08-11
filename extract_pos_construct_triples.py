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


def get_root(df):
    # extract the root of a sentence analyzed through NLP processing and stored in a df
    try:
        root_predicate = df.loc[df[3].str.contains('root')][1].values[0]
        root_position = int(df.loc[df[3].str.contains('root')][0].values[0])
    except:
        pass
    return root_predicate, root_position

def get_mwe(df, word_position):
    # determine multi word expressions
    if df.iloc[word_position + 1,3] == "fixed" or "flat" or "compound":
        mwe = df.iloc[word_position + 1, 1]
    return mwe


stanfordnlp.download('en')

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

# extract the root of that very data frame
root_elements = get_root(df)

# extract objects
objects_df = df.loc[df[3].str.contains('obj|obl')]
# if there is a reference in objects_df to the position of the root element, link it to the root
# df.loc[df[4] == root_elements[1]]

# extract subjects
subjects = []
# doing like it was done in next line will only create a copy of the dataframe
# subjects_ref = df.loc[df[3].str.contains('subj|cc|conj|flat')]
pos_subj = df.loc[df['dependency'].str.contains('subj')]['sentence_position'].values[0]
name_subj = df.loc[df['dependency'].str.contains('subj')]['text'].values[0]
# get mwes
nxt_pos_dpndcy = df.loc[df.loc[:,'sentence_position'] == str(int(pos_subj)+1),'dependency']
if nxt_pos_dpndcy.values[0] in ["fixed","flat","compound"]:
    mwe = df.loc[df.loc[:,'sentence_position'] == str(int(pos_subj)+1),'text']
    subject = name_subj + " " + mwe.values[0]
    subjects.append(subject)
if nxt_pos_dpndcy.values[0] not in ["fixed", "flat", "compound"]:
    subjects.append(name_subj)
# find additional subjects
if nxt_pos_dpndcy.values[0] in ["cc"]:
    upos_check = df.loc[df.loc[:, 'sentence_position'] == str(int(pos_subj) + 2), 'upos']
    if upos_check.values[0] in ["PROPN","PRON"]:
        second_subject = df.loc[df.loc[:, 'sentence_position'] == str(int(pos_subj) + 2), 'text']
        subjects.append(second_subject.values[0])

# to-do: make it a function, because the same is applicable to objects; extend functionality,
# currently MWE + multiple subjects like "Nick Fury and Maria Hill" does not work correctly
# compare output to open IE output

# to-do: add functionality to fallback to any other word in case that i.e. nsubj is missing or no object is there


from pycorenlp import StanfordCoreNLP
inst = StanfordCoreNLP('http://localhost:9000')

str_repr = ' '.join(word_text)

output = inst.annotate(str_repr, properties={
  'annotators': 'tokenize,ssplit,pos,depparse,parse,openie',
  'outputFormat': 'json'
  })

print(output['sentences'][0]['openie'])