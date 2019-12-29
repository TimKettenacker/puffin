# this code presents the grouped instances to the user on the command line; user picks the ones to persist
# it also replaces s,p,o in a copy of the persisted triples with a match to their concept in metadata_df,
# if applicable
# this code needs to be called from the terminal due to an issue with inquirer usage in PyCharm
import os
import inquirer
import pandas
from datetime import date

# prompt user to select a file to be uploaded
ask_for_file = [
  inquirer.List('x',
                message="Please select the file for review",
                choices=os.listdir(os.getcwd()),
            ),
]
selected_path = inquirer.prompt(ask_for_file)

# read file into pandas
triples4review_df = pandas.read_csv(selected_path.values()[0], delimiter=';')
triples_to_persist = []

# prompt user to select 0 or more of the grouped triples
for group_id in triples4review_df['group_id'].unique():
    choices = triples4review_df['triple'][triples4review_df['group_id'] == group_id]
    choices = choices.tolist()
    questions = [
        inquirer.Checkbox('triples_to_persist',
                          message="Please select only the individuals you want to persist",
                          choices=choices,
                          ),
    ]
    answers = inquirer.prompt(questions)
    for values in answers.values():
        for value in values:
            triples_to_persist.append(value)

reviewed_triples_df = pandas.DataFrame(triples_to_persist)
reviewed_triples_df = pandas.DataFrame(reviewed_triples_df[0].str.split(",").values.tolist())

# store results on disk
file_name1 = str(date.today().isoformat()) + "_reviewed_triples.csv"
reviewed_triples_df.to_csv(file_name1, sep=';', index=False)

# prompt user to upload the enrichment reference file in order to create a mapping
ask_for_file = [
  inquirer.List('y',
                message="Please select the reference file for enrichment",
                choices=os.listdir(os.getcwd()),
            ),
]
selected_path = inquirer.prompt(ask_for_file)
metadata_df = pandas.read_csv((selected_path.values()[0]), delimiter=";")
metadata_df = metadata_df.drop_duplicates(subset='Individual', keep='first')

individuals_mapped_to_concepts_df = reviewed_triples_df.copy()

# for all columns in the data frame, replace the values of individuals with their respective concepts
for z in range(0, len(individuals_mapped_to_concepts_df.columns)):
    for element in individuals_mapped_to_concepts_df[z]:
        try:
            index = metadata_df.index[metadata_df['Individual'].astype(str).str.contains(element)]
            concept = metadata_df['Concept'][index.values[0]]
            individuals_mapped_to_concepts_df[individuals_mapped_to_concepts_df == element] = concept
        except Exception as e:
            pass

# store the mapping of the individuals and their respective concepts
file_name2 = str(date.today().isoformat()) + "_mapped_concepts.csv"
individuals_mapped_to_concepts_df.to_csv(file_name2, sep=';', index=False)

print("Thanks! Your input has been processed and stored")