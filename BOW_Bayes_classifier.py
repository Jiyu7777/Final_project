from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import os
from os import listdir
import math
import numpy as np
from numpy import nan
import random
os.chdir('C:\\Users\\46726\\OneDrive\\Documents\\INFO523\\Final_project\\data')# for function merge to operate
df=pd.read_excel('SRL_reminding_4_11_2022.xlsx', sheet_name='Sheet1')

corpus=df['test_resp'].tolist()
cog=df['Jiyu_Cog'].tolist()
meta=df['Jiyu_Meta'].tolist()
#print(meta[0])
#print(len(cog),len(meta),len(corpus))
labels = []
labels = ['Cog' if cog[i] == 1.0 and meta[i] == 0.0
          else 'Meta' if cog[i] == 0.0 and meta[i] == 1.0
          else 'Both' if cog[i] == 1.0 and meta[i] == 1.0
          else 'Other' for i in range(len(cog))]

#print(labels.count('Both'))
# Create the vectorizer object
vectorizer = CountVectorizer()

# Fit and transform the corpus to create the bag of words representation
bow = vectorizer.fit_transform(corpus)

# Train a Naive Bayes classifier on the bag of words representation
clf = MultinomialNB()
clf.fit(bow, labels)

# Test the classifier on new data
df2=pd.read_excel('SRL_reminding.05_06_2021.xlsx', sheet_name='Sheet1')
new_data=df2['test_resp']
new_data = df2['test_resp'].fillna('none').tolist()
'''new_data[pd.isna(new_data)] = 'none'#replace nan 
new_data=new_data.tolist()'''
new_cog=df2['COG_final'].tolist()
new_meta=df2['META_final'].tolist()
#print(len(new_data),len(new_cog))

nan_indices = [i for i, x in enumerate(new_data) if pd.isna(x)]#check nan
#print(nan_indices)

new_bow = vectorizer.transform(new_data)
predictions = clf.predict(new_bow)

# Print the predictions
#print(predictions)
#Performance 
new_labels = ['Cog' if new_cog[i] == 1.0 and new_meta[i] == 0.0
              else 'Meta' if new_cog[i] == 0.0 and new_meta[i] == 1.0
              else 'Both' if new_cog[i] == 1.0 and new_meta[i] == 1.0
              else 'Other' for i in range(len(new_cog))]


correct_prediction=[]
for i in range(len(predictions)):
    if predictions[i]=='Cog' and new_cog[i]==1.0 and new_meta[i]==0.0:
        correct_prediction.append(1)
    elif predictions[i]=='Meta' and new_cog[i]==0.0 and new_meta[i]==1.0:
        correct_prediction.append(1)
    elif predictions[i]=='Both' and new_cog[i]==1.0 and new_meta[i]==1.0:
        correct_prediction.append(1)
    elif predictions[i]=='Other' and new_cog[i]==0.0 and new_meta[i]==0.0:
        correct_prediction.append(1)
    else:
        correct_prediction.append(0)
performance=(sum(correct_prediction)/len(correct_prediction))
print(performance)
###get the disagreements, this is critical if I want to know why there are disagreements 
