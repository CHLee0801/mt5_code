import pandas as pd  
import json 

with open("sample.json", "r") as st_json:
    st_python = json.load(st_json)
out_list = []
for idx, st in enumerate(st_python):
    out_list.append([idx, st[0], st[1]])

out = pd.DataFrame(out_list)
column_label = ['idx', 'input', 'output']

out.to_csv('train.csv', index=False, header=column_label)