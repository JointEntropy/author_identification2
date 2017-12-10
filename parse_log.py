import os
import shutil
import csv
# with open('log.csv') as log_csv:
#     articles = csv.DictReader(log_csv, delimiter=',', quotechar='/')
#     for article in articles:
#         print(article['author'])

import pickle
with open('bad_guys_shortcuts.pckl', 'rb') as fr:
    bad_guys_sc = pickle.load(fr)
print(bad_guys_sc)