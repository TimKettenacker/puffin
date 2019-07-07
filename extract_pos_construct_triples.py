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

annotated_words= doc.sentences[1].words
df = pandas.DataFrame()

word_index = []
word_lemma = []
word_upos = []
word_dependency_relation = []
word_governor = []

for u in range(1, len(annotated_words)):
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

# create a subset of the dataframe containing only the elements for triple generation
elements_df = df.loc[df[3].str.contains('nsubj|flat|conj|root|compound|obj|cc')]

# extract subject(s)
subjects = []
