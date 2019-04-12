#!/usr/bin/env python
# coding: utf-8

# <center><h1>US Presidents Inaugural Speech Analysis</h1></center>
# <center><img src="hqdefault.jpg"></center>
# 
# ## Introduction
# 
# The data set is made up of 58 files each containing the speech of inaugural speech of all American presidents.
# 
# 
# 

# In[188]:


# !pip install -r requirement.txt


# ## Imports and function definitions
# 

# In[189]:


import glob
import re
import string

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.cluster.hierarchy import dendrogram
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

stop_words = list(set(stopwords.words('english') + list(ENGLISH_STOP_WORDS)))
lemmatizer = WordNetLemmatizer()

def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def get_info():
    global index, values, label, val
    for index in range(fit.n_clusters):
        values = indices[int(index)]
        label = [labels[val] for val in values]
        print(50 * "-")
        print("Cluster %s Information" % index)
        for val in set(label):
            print("Value:", str(val), "Count:", str(label.count(val)))

def get_data_df(file_path):
    df = pd.DataFrame(columns=['year', 'prez_name', 'speech'])
    for file in glob.glob("speeches/*.txt"):
        prez_name = file.split('\\')[1].split('-')[1].split('.')[0]
        year = file.split('\\')[1].split('-')[0]
        with open(file, 'r') as f:
            df.loc[len(df)] = [year, prez_name, f.read()]
    return df

def count_vec2df(wm, feat_names):
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names, columns=feat_names)
    return (df)

def preprocess_text(text):
    text = text.lower()  # lower case
    text = re.sub(r'[' + string.punctuation + ']+', ' ', text)  # strip punctuation
    text = re.sub(r'\s+', ' ', text)  #remove double spacing
    text = re.sub(r'([0-9]+)', '', text)  # remove numbers
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split(' ')
                     if word not in stop_words and len(word) > 2])
    return text

def tokenize(text):
    return [token for token in simple_preprocess(text)]


# In[ ]:





# In[190]:


speeches_df = get_data_df("speeches/*.txt")
speeches_df['processed_speech'] = speeches_df.speech.map(preprocess_text) 

all_text = " ".join(speeches_df.processed_speech.tolist())
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(all_text)

# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show()


# In[240]:


theme_words = 'military,war'
words = theme_words.split(',')
speeches_df['theme_words_percent_usage']=speeches_df.processed_speech.map(lambda x: sum([x.count(" "+preprocess_text(word.lower())) for word in words])/len(x.split())*100)
speeches_df.plot(kind = 'line', x='year', y='theme_words_percent_usage', legend=False, figsize =(20, 6), linestyle="None", marker="o", )
plt.show()


# In[192]:


doc_term_vect = CountVectorizer(ngram_range=(1, 2),
                       min_df=2, # only keep terms that appear in at least 2 documents
                       max_df=0.5, # ignore terms that appear in more than 50% of the documents
                       token_pattern=r'\w{2,}' #vectorize 2-character words or more
                       )
bag_of_words = doc_term_vect.fit_transform(speeches_df.processed_speech.tolist())
doc_term_mtx_df = count_vec2df(bag_of_words, doc_term_vect.get_feature_names())
doc_term_mtx_df.to_csv("project4_dtm.csv")

tf_id_vect = TfidfVectorizer(ngram_range=(1, 2),
                             token_pattern=r'\w{2,}'
                            )
tf_id_mtx = tf_id_vect.fit_transform(speeches_df.processed_speech.tolist())
tf_id_mtx_df = count_vec2df(tf_id_mtx, tf_id_vect.get_feature_names())
tf_id_mtx_df.to_csv("project4_tf_id_mtx.csv")


# In[193]:


scores = []
sum_of_squared_distances = []
print("get the number of k...")
for k in range(2, 30):
    km = KMeans(n_clusters=k).fit(bag_of_words)
    sc = metrics.calinski_harabaz_score(bag_of_words.toarray(), km.labels_)
    scores.append(sc)
    sum_of_squared_distances.append(km.inertia_)
    
plt.plot(scores)
plt.ylabel('some numbers')
plt.show()


# In[250]:


feature_vector = bag_of_words
feature_vector = tf_id_mtx
k=15

fit = KMeans(n_clusters=k).fit(feature_vector)
speeches_df['km_classes']  = fit.labels_
# labels = fit.labels_
# indices = {i: np.where(fit.labels_ == i)[0] for i in range(fit.n_clusters)}
# get_info()

# hac = AgglomerativeClustering(n_clusters=k)
# fit_hac = hac.fit(feature_vector.toarray())
# speeches_df['hac_classes'] = fit_hac.labels_
# labels = fit.labels_
# # indices = {i: np.where(fit.labels_ == i)[0] for i in range(fit_hac.n_clusters)}
# # get_info()

# plt.figure(num=None, figsize=(20, 6), dpi=80, facecolor='w', edgecolor='k')
# plot_dendrogram(fit_hac)
# plt.show()

speeches_df.head()


# In[254]:


documents = speeches_df.processed_speech.tolist()
texts = [tokenize(document) for document in documents]
all_text = " ".join(speeches_df.processed_speech.tolist())
frequency = Counter(all_text.split())
texts = [[token for token in text if frequency[token] > 10] for text in texts]
 
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# fit LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus,
                           id2word=dictionary,
                           num_topics=NUM_TOPICS,
                           passes=20)

for topic in ldamodel.print_topics(num_words=20):
    print(topic)

# vis_data = gensimvis.prepare(ldamodel, corpus, dictionary)
# pyLDAvis.display(vis_data)


# In[259]:


# corpus


# In[197]:


# doc_term_vect.vocabulary_['immigration']
# # speeches_df
# tf_id_vect.idf_
# print(list(zip(tf_id_vect.get_feature_names(), tf_id_vect.idf_)))
# all_text1= " ".join(speeches_df.processed_speech.tolist())
all_text = " ".join(speeches_df.speech.tolist())
# speeches_topics.
# all_text


# In[ ]:




