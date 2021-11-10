import pandas as pd

from sklearn.metrics import accuracy_score


df1 = pd.read_csv("../Dataset/movie_review.csv")
data1 = df1['label']

df2 = pd.read_csv("sentiment_output.csv")
data2 = df2['overall_sentiment']

ascore = accuracy_score(data1, data2)
print("\n\n>> Accuracy Score: ", ascore * 100)
