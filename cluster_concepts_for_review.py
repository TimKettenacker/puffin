# the code replaces s,p,o in persisted triples with a match to their concept in metadata_df, if applicable
# this code needs to be called from the terminal due to an issue with inquirer usage in PyCharm
import os
import inquirer
import pandas
from datetime import date
import csv

# prompt user to select a file containing individuals to be uploaded
ask_for_file = [
  inquirer.List('x',
                message="Please select the file to be enriched with metadata",
                choices=os.listdir(os.getcwd()),
            ),
]
selected_path = inquirer.prompt(ask_for_file)
reader = csv.reader(open(selected_path.values()[0]), delimiter=";")

reviewed_triples = list(reader)
reviewed_triples = [item for sublist in reviewed_triples for item in sublist]
reviewed_triples_df = pandas.DataFrame(reviewed_triples)
reviewed_triples_df = pandas.DataFrame(reviewed_triples_df[0].str.split(",").values.tolist())

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

# for all columns in the data frame, replace the values of individuals with their respective concepts
for z in range(0, len(reviewed_triples_df.columns)):
    for element in reviewed_triples_df[z]:
        try:
            index = metadata_df.index[metadata_df['Individual'].astype(str).str.contains(element)]
            concept = metadata_df['Concept'][index.values[0]]
            reviewed_triples_df[reviewed_triples_df == element] = concept
        except Exception as e:
            pass

file_name = str(date.today().isoformat()) + "_mapped_concepts.csv"
reviewed_triples_df.to_csv(file_name, sep=';', index=False)

print("Thank you! A mapping is now available")