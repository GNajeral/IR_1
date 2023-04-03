#!/usr/bin/env python
# coding: utf-8

# # IREI: Profile-based retrieval
# ### Víctor Morcuende Castell and Guillermo Nájera Lavid
# #### Course 2022-2023

# ### Preprocessing Phase

# In[1]:


import nltk

nltk.download('all')


# In[2]:


# Read the data
import pandas as pd

train_data = pd.read_csv('dataset/BBC News Train.csv')
test_data = pd.read_csv('dataset/BBC News Test.csv')

# Transform the data into a single dataset
data = pd.concat([train_data,test_data])
data.to_csv('dataset/data.csv', index=False)

# Remove duplicated data
data = data.drop_duplicates(subset=['Text','Category'])
data.head(10)


# In[3]:


data.shape


# In[4]:


data.groupby(['Category']).size().sort_values(ascending=True)


# In[5]:


data.groupby(['Category']).size().sort_values(ascending=True).plot(kind='barh', figsize=(10, 6))


# In[6]:


# Remove all punctuations from the text
import string as st

def remove_punct(text):
    return ("".join([ch for ch in text if ch not in st.punctuation]))

data['removed_punc'] = data['Text'].apply(lambda x: remove_punct(x))
data.head()


# In[7]:


# Convert text to lower case tokens
import re

def tokenize(text):
    text = re.split('\s+' ,text)
    return [x.lower() for x in text]

data['tokens'] = data['removed_punc'].apply(lambda msg : tokenize(msg))
data.head()


# In[8]:


# Remove tokens of length less than 3
def remove_small_words(text):
    return [x for x in text if len(x) > 3 ]

data['larger_tokens'] = data['tokens'].apply(lambda x : remove_small_words(x))
data.head()


# In[9]:


# Remove stopwords by using NLTK corpus list
def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]

data['clean_tokens'] = data['larger_tokens'].apply(lambda x : remove_stopwords(x))
data.head()


# In[10]:


# Apply lemmatization on tokens
from nltk import WordNetLemmatizer

def lemmatize(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]

data['lemma_words'] = data['clean_tokens'].apply(lambda x : lemmatize(x))
data.head()


# In[11]:


# Create sentences to get clean text as input for vectors
def return_sentences(tokens):
    return " ".join([word for word in tokens])

data['clean_text'] = data['lemma_words'].apply(lambda x : return_sentences(x))
data.head()


# ### Model and Evaluation Phase

# In[12]:


import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity


# In[13]:


# Balancing the dataset to have the same number of documents for each query
from sklearn.utils import resample

def balance_data(data, category_col):
    categories = data[category_col].unique()
    min_category_count = data[category_col].value_counts().min()

    balanced_data = []

    for category in categories:
        category_data = data[data[category_col] == category]
        category_data_balanced = resample(category_data, replace=False, n_samples=min_category_count, random_state=42)
        balanced_data.append(category_data_balanced)

    return pd.concat(balanced_data)


# In[14]:


data = balance_data(data, 'Category')
balanced_data = data[['clean_text', 'Category']]
X_train, X_test, y_train, y_test = train_test_split(balanced_data['clean_text'], balanced_data['Category'], test_size=0.2, random_state=42)
balanced_data.groupby(['Category']).size().sort_values(ascending=True)


# In[15]:


balanced_data.groupby(['Category']).size().sort_values(ascending=True).plot(kind='barh', figsize=(10, 6))


# In[16]:


vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(balanced_data['clean_text'])


# In[17]:


topics = {
    # Sports
    'sports': ["sports", "championship", "soccer", "race", "football", "tennis", "baseball", "hockey", "basketball", "athletics", "rugby", "swimming", "golf", "cycling", "cricket", "marathon", "gymnastics", "boxing", "volleyball", "badminton", "fencing", "wrestling", "snowboarding", "skiing", "horse-racing", "archery", "table-tennis", "e-sports", "fitness", "olympics"],
    
    # Business
    'business': ["business", "finance", "stocks", "economy", "investment", "entrepreneurship", "corporation", "market", "trade", "revenue", "profit", "startup", "loss", "growth", "acquisition", "tax", "debt", "funding", "venture", "capital", "inflation", "interest", "dividends", "corporate", "management", "banking", "insurance", "real-estate", "franchise", "supply-chain"],
    
    # Entertainment
    'entertainment': ["entertainment", "movies", "music", "television", "celebrities", "awards", "festivals", "concert", "theater", "comedy", "drama", "action", "romance", "animation", "documentary", "dance", "art", "literature", "photography", "sculpture", "painting", "opera", "magic", "circus", "museum", "exhibition", "actor", "actress", "singer", "culture"],
    
    # Politics
    'politics': ["politics", "government", "elections", "policy", "democracy", "president", "parliament", "vote", "prime-minister", "congress", "senate", "international", "relations", "diplomacy", "referendum", "constitution", "legislation", "political-party", "campaign", "debate", "rights", "protest", "activism", "military", "intelligence", "treaty", "embassy", "visa", "immigration", "trade-agreements"],
    
    # Tech
    'tech': ["tech", "technology", "innovation", "gadgets", "smartphone", "artificial-intelligence", "robotics", "software", "hardware", "computer", "internet", "cybersecurity", "virtual-reality", "augmented-reality", "machine-learning", "data-science", "blockchain", "cryptocurrency", "internet-of-things", "cloud-computing", "big-data", "quantum-computing", "networking", "operating-system", "mobile-apps", "programming", "research", "drones", "3D-printing", "wearables"]
}


# In[18]:


users = [
    {'id': 1, 'interests': ['sports']},
    {'id': 2, 'interests': ['business']},
    {'id': 3, 'interests': ['entertainment']},
    {'id': 4, 'interests': ['politics']},
    {'id': 5, 'interests': ['tech']},
    {'id': 6, 'interests': ['sports', 'business']},
    {'id': 7, 'interests': ['entertainment', 'politics']},
    {'id': 8, 'interests': ['tech', 'sports']},
    {'id': 9, 'interests': ['business', 'entertainment']},
    {'id': 10, 'interests': ['politics', 'tech', 'business']}
]


# In[19]:


vec_user1 = vectorizer.transform([" ".join(users[0]['interests'])])
vec_user2 = vectorizer.transform([" ".join(users[1]['interests'])])
vec_user3 = vectorizer.transform([" ".join(users[2]['interests'])])
vec_user4 = vectorizer.transform([" ".join(users[3]['interests'])])
vec_user5 = vectorizer.transform([" ".join(users[4]['interests'])])
vec_user6 = vectorizer.transform([" ".join(users[5]['interests'])])
vec_user7 = vectorizer.transform([" ".join(users[6]['interests'])])
vec_user8 = vectorizer.transform([" ".join(users[7]['interests'])])
vec_user9 = vectorizer.transform([" ".join(users[8]['interests'])])
vec_user10 = vectorizer.transform([" ".join(users[9]['interests'])])


# In[20]:


lista_vecs = []
lista_vecs.append(vec_user1)
lista_vecs.append(vec_user2)
lista_vecs.append(vec_user3)
lista_vecs.append(vec_user4)
lista_vecs.append(vec_user5)
lista_vecs.append(vec_user6)
lista_vecs.append(vec_user7)
lista_vecs.append(vec_user8)
lista_vecs.append(vec_user9)
lista_vecs.append(vec_user10)


# In[34]:


import random

for i in range(10):
  min_length = min(len(balanced_data['clean_text']), document_vectors.shape[0])
  random_index = random.randint(0, min_length - 1) 
  incoming_doc_vector = document_vectors[random_index]
  list_sim = []

  profiles = []

  for j in range(len(lista_vecs)):
    similarities = cosine_similarity(incoming_doc_vector, lista_vecs[j]) 
    if similarities[0][0] > 0.0:
      profiles.append(j+1)
      list_sim.append(similarities[0][0])
  
  print("For document",i+1, ":", balanced_data['clean_text'].iloc[random_index])
  print("Categorized as: "+ balanced_data['Category'].iloc[random_index]+ ' topic.')
  print("The user who are interested in this document are", profiles)
  print()

  print("RANKING")
  ranking = pd.DataFrame()
  ranking["Users"] = profiles
  ranking["Score"] = list_sim
  ranking = ranking.sort_values('Score', ascending=False)
  print(ranking)
  print()

