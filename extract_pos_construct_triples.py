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


stanfordnlp.download('en')

nlp = stanfordnlp.Pipeline()
doc = nlp(plot_dict[urls[22]])

# data objects need to be extracted for ontologies, see https://stanfordnlp.github.io/stanfordnlp/data_objects.html
# dependency_relation is a key element (word to grammar relationships); while 'nsubj', 'nsubj:pass', 'compound' and
# 'flat' characterize subjects and 'obj' the object, relationship values are often indicated by the 'root' and sometimes
# the 'conj' dependency. Yet, there is no exhaustive list of possible dependency values, but the alphabetical listing of
# the Universal Dependency project is expanded on https://universaldependencies.org/u/dep/index.html
# data object governor means a word that another word is dependent on

# map data objects
# try recognizing entities and map relations

annotated_words = doc.sentences[0].words
df = pandas.DataFrame()

word_index = []
word_lemma = []
word_upos = []
word_dependency_relation = []
word_governor = []

for u in range(0, len(annotated_words)):
    word_index.append(annotated_words[u].index)
    word_lemma.append(annotated_words[u].lemma)
    word_upos.append(annotated_words[u].upos)
    word_dependency_relation.append(annotated_words[u].dependency_relation)
    word_governor.append(annotated_words[u].governor)

df[0] = word_index
df[1] = word_lemma
df[2] = word_upos
df[3] = word_dependency_relation
df[4] = word_governor

# clean up data frame
df = df.loc[~df[2].str.contains('PUNCT')]

# extract the root of that very data frame
root_elements = get_root(df)

# extract objects
objects_df = df.loc[df[3].str.contains('obj')]
# if there is a reference in objects_df to the position of the root element, link it to the root
# df.loc[df[4] == root_elements[1]]

# extract subjects
subjects = []
subjects_df = df.loc[df[3].str.contains('nsubj|cc|conj|flat')]
nsubj_position = int(subjects_df.loc[subjects_df[3].str.contains('nsubj')][0].values)
nsubj_name = subjects_df.loc[subjects_df[3].str.contains('nsubj')][1].values[0]
subjects.append(nsubj_name)
# try to identify additional subjects
if len(subjects_df) > 1:
    try:
        position_other_term = int(subjects_df.loc[subjects_df[4] == nsubj_position][0].values[0])
        if ['CCONJ' in subjects_df[nsubj_position:position_other_term].values] == [True]:
            second_subj_name = subjects_df.loc[subjects_df[4] == nsubj_position][1].values[0]
            subjects.append(second_subj_name)
    except:
        pass

# to-do: add functionality that includes sur- last name, like Nick Fury, where last name is either flat or conj
# do the same also if a second person is detected
