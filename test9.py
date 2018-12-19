import pandas as pd
import os


path = "corpus/AP/1T/LR"
file_result = path + "/resultat.csv"

result = []

for file in os.listdir(path):
    if file.endswith(".csv"):
        f =  open(path + "/"+file,"r")
        my_df = pd.read_csv(f, index_col=0, sep="\t")
        result.append(my_df)



frame = pd.concat(result, axis = 1, ignore_index = True)


frame.to_csv(file_result, sep='\t')