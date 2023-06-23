from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import tree

# gather data
music_dt  =pd.read_csv('music.csv')

# display the data
music_dt

# prepare 2 groups
X=music_dt.drop(columns=['genre']) # sample features
Y=music_dt['genre'] # sample output

model = DecisionTreeClassifier()
model.fit(X,Y) # load features and sample data

tree.export_graphviz(model,out_file='music_rec.dot',feature_names=['age','gender'],class_names=sorted(Y.unique()),label='all',rounded=True,filled=True)
