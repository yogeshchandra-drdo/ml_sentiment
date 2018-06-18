import numpy as np
import pandas as pd

print("Sentiment Analysis Project")
print("--------------------------")

#f = open("amazon_cells_labelled.csv")
#f.readline() # skip the header
#data = np.loadtxt(f)

df = pd.read_csv("amazon_cells_labelled.csv", header=1)
for row in df:
    print(row)

#df1 = df.iloc[:,0:1]
#df1 = df[['sentence']]
df1 = df[df.columns[0:2]]
print(df1)



