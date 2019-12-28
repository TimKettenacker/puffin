# this code presents the grouped instances to the user on the command line; user picks the ones to persist
# this code needs to be called from the terminal due to an issue with inquirer usage in PyCharm

import os
import inquirer
import pandas
from datetime import date
import csv

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

# write reviewed triples to disk
file_name = str(date.today().isoformat()) + "_reviewed_triples.csv"

with open(file_name, "w") as f:
    writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(triples_to_persist)

print("Thanks! Your input has been stored")