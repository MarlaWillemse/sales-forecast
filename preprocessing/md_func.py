import os
from root import *
import pandas as pd

preds = pd.read_csv(root+"/data/processed/predictions.csv")

preds['Predicted Volume'] = preds['Future_Volume']
preds = preds.drop('Future_Volume', axis=1)

print()

preds_md = preds.to_markdown()

text_file = open(root+"/data/processed/preds_table.md", "w")
n = text_file.write(preds_md)
text_file.close()

file = open(root+"/reports/report.md", "a")
file.write(preds_md)
file.close()

# TODO: avoid appending each time file is run
