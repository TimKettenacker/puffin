# variable "triples" contains the instance data, before creation of RDF triples can be commenced, a meta schema
# must be generated to set the namespaces for URIRefs and BNodes with the help of python's rdflib or similar,
# in order to map values like "Maria Hill" to schema.org/person etc.
# attempting to infer semantic topics with cosine similarity and tf idf did not prove to be useful for the task
# at hand, because the only issue with the triples at this stage are superfluous tokens, so instead this code
# groups the triples together to let a user decide which ones to keep and complete the metadata model
# instance data can be mapped to owl after ontology is created, "triples" variable could be added as individuals in
# https://owlready2.readthedocs.io/en/latest/ - can read the RDF triples and other ontologies / vocabulary,
# is able to alter and manage them in an integrated quad store

import fuzzywuzzy
from fuzzywuzzy import fuzz

triples_df['triple'] = triples_df['Subject'].map(str) + "," + triples_df['Predicate'] + "," + triples_df['Object']
triples = triples_df['triple'].to_list()

match_groups = []
id = 0
for word in triples:
    added_to_existing = False
    for merged in match_groups:
        try:
            for potentially_similar in merged[0]:
                if (fuzz.partial_ratio(word, potentially_similar) > 75) & (fuzz.token_set_ratio(word, potentially_similar) > 75):
                    merged[0].add(word)
                    added_to_existing = True
                    break
            if added_to_existing:
                break
        except Exception as e:
            logger.error(e)
            pass
    if not added_to_existing:
        id += 1
        match_groups.append([set([word]), id])


triples_df['group_id'] = ""
for index, row in triples_df.iterrows():
    for kk in range(0, len(match_groups)):
        if (match_groups[kk][0].__contains__(row['triple']) == True):
            triples_df['group_id'][index] = match_groups[kk][1]