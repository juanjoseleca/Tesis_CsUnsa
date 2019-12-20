import pandas as pd
import re
from wordcloud import WordCloud
import os
from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from classifier import *
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF

#Graficar 
def plot_10_most_common_words(count_data, count_vectorizer):
    
    words = count_vectorizer.get_feature_names()
    
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()



def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\n Topicos #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
                        
                        

dataSet = pd.read_csv('Data_silabuz.csv',sep=',')

nameColum='comentarios'
clf = SentimentClassifier()
dataSet[nameColum] = dataSet[nameColum].map(lambda x: re.sub('[,\.!?\n]', '', x))
dataSet["categoria"] = np.nan
dataSet["rating"] = np.nan
vectorizer = CountVectorizer()
sns.set_style('whitegrid')

count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(dataSet[nameColum])

number_topics = 10
number_words = 10

lda = LDA(n_components=number_topics)
lda.fit(count_data)

print("Aplicando LDA")
print_topics(lda, count_vectorizer, number_words)
print(lda.transform(count_data[:2]))
total=len(count_data.todense())
print("-->",total)
converted=lda.transform(count_data)
sentiment=[]
for i in range(total):
    max_element=max(converted[i])
    position=[i for i, j in enumerate(converted[i]) if j == max_element][0]
    dataSet.loc[dataSet.index[[i]], 'categoria'] = position
    #print("COMENTARIO: ",dataSet.loc[dataSet.index[[i]].values[0], 'comentarios'])
    #print(" valoracion: ",clf.predict(dataSet.loc[dataSet.index[[i]].values[0], 'comentarios']))
    sentiment.append(clf.predict(dataSet.loc[dataSet.index[[i]].values[0], 'comentarios']))
lower, upper = 1, 5
data_normalizada = [lower + (upper - lower) * x for x in sentiment]
for i in range(total):
    dataSet.loc[dataSet.index[[i]], 'rating'] = data_normalizada[i]
dataSet.to_csv("tabla1.csv")
matriz=dataSet.pivot_table(index='ID', columns='categoria', values='rating',aggfunc='mean', fill_value=0)
matriz.to_csv("ratings_i.csv")
matriz_pmf=[]
for i, row in matriz.iterrows():
    for j in range(number_topics):
        matriz_pmf.append([i,int(row.index[j]),row[row.index[j]]])
matriz_pmf=np.array(matriz_pmf)
pmf = PMF()
pmf.set_params({"num_feat": 10, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 20, "num_batches": 400,"batch_size": 100})
train, test = train_test_split(matriz_pmf, test_size=0.2)
pmf.fit(train, test)
recomendaciones = pd.DataFrame(pmf.predict_all())
recomendaciones.to_csv("recomendaciones.csv")

#print(matriz_pmf)