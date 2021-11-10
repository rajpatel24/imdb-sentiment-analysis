import pandas as pd

from sklearn.metrics import accuracy_score

df2 = pd.read_csv("result.csv")
data1 = df2['label']
data2 = df2['sentiment_score']

ascore = accuracy_score(data1, data2)
print("\n\n>> Accuracy Score: ", ascore * 100)
