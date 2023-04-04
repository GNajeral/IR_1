#!/usr/bin/env python
# coding: utf-8

# # IREI: Profile-based retrieval
# ### Víctor Morcuende Castell and Guillermo Nájera Lavid
# #### Course 2022-2023

# ### Preprocessing Phase

# In[26]:


import nltk

nltk.download('all')


# In[27]:


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


# In[28]:


data.shape


# In[29]:


data.groupby(['Category']).size().sort_values(ascending=True)


# In[30]:


data.groupby(['Category']).size().sort_values(ascending=True).plot(kind='barh', figsize=(10, 6))


# In[31]:


# Remove all punctuations from the text
import string as st

def remove_punct(text):
    return ("".join([ch for ch in text if ch not in st.punctuation]))

data['removed_punc'] = data['Text'].apply(lambda x: remove_punct(x))
data.head()


# In[32]:


# Convert text to lower case tokens
import re

def tokenize(text):
    text = re.split('\s+' ,text)
    return [x.lower() for x in text]

data['tokens'] = data['removed_punc'].apply(lambda msg : tokenize(msg))
data.head()


# In[33]:


# Remove tokens of length less than 3
def remove_small_words(text):
    return [x for x in text if len(x) > 3 ]

data['larger_tokens'] = data['tokens'].apply(lambda x : remove_small_words(x))
data.head()


# In[34]:


# Remove stopwords by using NLTK corpus list
def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]

data['clean_tokens'] = data['larger_tokens'].apply(lambda x : remove_stopwords(x))
data.head()


# In[35]:


# Apply lemmatization on tokens
from nltk import WordNetLemmatizer

def lemmatize(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]

data['lemma_words'] = data['clean_tokens'].apply(lambda x : lemmatize(x))
data.head()


# In[36]:


# Create sentences to get clean text as input for vectors
def return_sentences(tokens):
    return " ".join([word for word in tokens])

data['clean_text'] = data['lemma_words'].apply(lambda x : return_sentences(x))
data.head()


# ### User's Creation and Documents' Encoding

# In[37]:


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


# In[38]:


data = balance_data(data, 'Category')
balanced_data = data[['clean_text', 'Category']]
balanced_data.groupby(['Category']).size().sort_values(ascending=True)


# In[39]:


balanced_data.groupby(['Category']).size().sort_values(ascending=True).plot(kind='barh', figsize=(10, 6))


# In[40]:


from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5)
#vectorizer = TfidfVectorizer(norm='l2', use_idf=False)
vectorizer = TfidfVectorizer(stop_words='english')
document_vectors = vectorizer.fit_transform(balanced_data['clean_text'])


# In[41]:


topics = {
    # Sport
    'sport': ['sport', 'game', 'year', 'first', 'player', 'world', 'england', 'time', 'play', 'match', 'team', 'side',
              'second', 'champion', 'good', 'three', 'week', 'ireland', 'final', 'coach', 'injury', 'season', 'club',
              'france', 'wale', 'people', 'rugby', 'open', 'great', 'nation', 'month', 'point', 'united', 'title', 
              'minute', 'start', 'victory', 'chance', 'chelsea', 'international', 'played', 'scotland', 'best', 'home', 
              'league', 'championship', 'five', 'olympic', 'face', 'goal', 'playing', 'record', 'arsenal', 'country',
              'decision', 'place', 'test', 'manager', 'race', 'break', 'return', 'grand', 'beat', 'european', 'four',
              'away', 'former', 'service', 'third'],
    
    # Business
    'business': ['business', 'year', 'company', 'firm', 'market', 'country', 'government', 'sale', 'bank', 'price', 
                 'economy', 'growth', 'month', 'share', 'economic', 'world', 'rate', 'people', 'time', 'analyst', 
                 'chief', 'first', 'deal', 'profit', 'dollar', 'rise', 'china', 'euro', 'offer', 'cost', 'plan', 
                 'executive', 'group', 'make', 'three', 'week', 'figure', 'financial', 'minister', 'report', 
                 'investment', 'stock', 'many', 'india', 'yukos', 'interest', 'high', 'state', 'debt', 'demand', 
                 'european', 'well', 'trade', 'foreign', 'million', 'move', 'strong', 'director', 'president', 'good',
                 'industry', 'number', 'quarter', 'budget', 'fall', 'former', 'news', 'work', 'money', 'need', 'investor'],
    
    # Entertainment
    'entertainment': ['entertainment', 'film', 'year', 'best', 'award', 'people', 'show', 'music', 'star', 'time', 
                      'number', 'actor', 'band', 'director', 'world', 'album', 'like', 'company', 'sale', 'million', 
                      'government', 'oscar', 'song', 'chart', 'home', 'record', 'movie', 'role', 'actress', 'place',
                      'play', 'right', 'week', 'single', 'game', 'group', 'life', 'singer', 'work', 'prize', 'country',
                      'industry', 'good', 'festival', 'nomination', 'party', 'money', 'child', 'office', 'comedy', 'rock', 
                      'winner', 'series', 'book', 'woman', 'producer', 'news', 'love', 'performance', 'musical'],
    
    # Politics
    'politics': ['politics', 'labour', 'people', 'government', 'party', 'election', 'year', 'blair', 'minister', 'tory',
                 'plan', 'time', 'brown', 'lord', 'country', 'public', 'home', 'issue', 'leader', 'right', 'game', 'secretary', 
                 'general', 'service', 'prime', 'week', 'world', 'change', 'campaign', 'like', 'conservative', 'bill', 'spokesman',
                 'chancellor', 'police', 'report', 'child', 'claim', 'council', 'power', 'vote', 'need', 'liberal', 'democrat', 
                 'case', 'policy', 'member', 'court', 'problem', 'european', 'group', 'former', 'house', 'help', 'local', 'system',
                 'decision', 'school', 'kennedy', 'news', 'office', 'place', 'state'],
    
    # Tech
    'tech': ['tech', 'technology', 'people', 'year', 'game', 'mobile', 'phone', 'service', 'firm', 'user', 'time', 'music', 'first',
             'company', 'computer', 'software', 'system', 'world', 'like', 'digital', 'number', 'million', 'network', 'used', 'player',
             'market', 'work', 'online', 'consumer', 'microsoft', 'site', 'internet', 'device', 'month', 'broadband', 'website', 'video',
             'gadget', 'show', 'data', 'home', 'information', 'medium', 'machine', 'search', 'security', 'european', 'content', 'research',
             'report', 'group', 'news', 'help', 'virus', 'industry', 'problem', 'email', 'mean', 'program', 'message', 'play', 'camera', 
             'different', 'three', 'apple', 'europe', 'offer', 'sale']
}


# In[42]:


users = [
    {'id': 1, 'interests': topics['sport']},
    {'id': 2, 'interests': topics['business']},
    {'id': 3, 'interests': topics['entertainment']},
    {'id': 4, 'interests': topics['politics']},
    {'id': 5, 'interests': topics['tech']},
    {'id': 6, 'interests': topics['sport'] + topics['business']},
    {'id': 7, 'interests': topics['entertainment'] + topics['politics']},
    {'id': 8, 'interests': topics['tech'] + topics['sport']},
    {'id': 9, 'interests': topics['business'] + topics['entertainment']},
    {'id': 10, 'interests': topics['politics'] + topics['tech'] + topics['business']}
]


# Simple way of creating the users

# In[43]:


# user_vectors = []
# for user in users:
#     interests = " ".join(user['interests'])
#     vector = vectorizer.transform([interests])
#     user_vectors.append(vector)

# lista_vecs = [user_vectors[i] for i in range(len(user_vectors))]


# Creating the users by using the mean/max function

# In[44]:


# import numpy as np
# from scipy.sparse import csr_matrix

# def aggregate_vectors(vectors):
#     return np.mean(vectors, axis=0)

# def max_aggregate_vectors(vectors):
#     return np.max(vectors, axis=0)

# user_vectors = []
# for user in users:
#     topic_vectors = []
#     for topic in user['interests']:
#         topic_vector = vectorizer.transform([topic]).toarray()
#         topic_vectors.append(topic_vector)
#     user_vector = max_aggregate_vectors(topic_vectors)
#     user_vectors.append(user_vector)

# user_vectors_sparse = [csr_matrix(user_vector) for user_vector in user_vectors]

# lista_vecs = user_vectors_sparse


# #### Creating the users by using the Weighted Topic Frequency (WTF) method
# 
# This a creative approach for constructing user vectors that takes into account the uniqueness of each topic for the user.
# 
# 1. Calculate the term frequency (TF) for each word in the user's interests.
# 2. Calculate the inverse topic frequency (ITF) for each word across all topics.
# 3. Calculate the Weighted Topic Frequency (WTF) for each word by multiplying its TF by its ITF.
# 4. Create the user vector by using the WTF values for each word in the user's interests.
# 
# Here's a step-by-step explanation:
# 
# 1. Term Frequency (TF): Count the frequency of each word in the user's interests and normalize it by the total number of words in the user's interests.
# 
# 2. Inverse Topic Frequency (ITF): For each word in the user's interests, calculate its presence in all topics. Then, compute the inverse of this presence (total number of topics / number of topics containing the word). This will give higher weights to words that are more unique to a user's interests.
# 
# 3. Weighted Topic Frequency (WTF): Multiply the TF and ITF for each word to obtain the WTF value. This will emphasize words that are both frequent in the user's interests and unique to their topics.
# 
# 4. User vector creation: Use the WTF values for each word in the user's interests to create the user vector. This can be done by transforming the user's interests (with WTF values) using the vectorizer.transform() function.

# In[45]:


import numpy as np
from collections import Counter

def calculate_tf(user_interests):
    word_count = Counter(user_interests)
    total_words = len(user_interests)
    tf = {word: count / total_words for word, count in word_count.items()}
    return tf

def calculate_itf(user_interests, topics):
    num_topics = len(topics)
    topic_presence = {word: 0 for word in user_interests}
    
    for topic_words in topics.values():
        for word in set(user_interests):
            if word in topic_words:
                topic_presence[word] += 1
    
    itf = {word: np.log(num_topics / presence) for word, presence in topic_presence.items()}
    return itf

def calculate_wtf(user_interests, topics):
    tf = calculate_tf(user_interests)
    itf = calculate_itf(user_interests, topics)
    wtf = {word: tf[word] * itf[word] for word in user_interests}
    return wtf

user_vectors = []
for user in users:
    wtf = calculate_wtf(user['interests'], topics)
    weighted_interests = " ".join([word for word, weight in wtf.items() for _ in range(int(weight * 100))])
    user_vector = vectorizer.transform([weighted_interests])
    user_vectors.append(user_vector)

lista_vecs = [user_vectors[i] for i in range(len(user_vectors))]


# In[46]:


for user in lista_vecs:
    print(user)


# In[47]:


from sklearn.metrics.pairwise import cosine_similarity

predictions = []
predictions2 = []
for i in range(0, len(lista_vecs)):
    match = 0
    best_similarity = -1
    for j in range(0, document_vectors.shape[0]):
        document = document_vectors[j]
        similarity = cosine_similarity(document, lista_vecs[i])
        if similarity > best_similarity:
            best_similarity = similarity
            match = j
    predictions.append(balanced_data.iloc[match]['Category'])
    predictions2.append(balanced_data.iloc[match]['clean_text'])


# In[48]:


correct_predictions = 0
for category, text, user in zip(predictions, predictions2, users):
    print()
    print("User: ", user['id'])
    print("Category Predicted: ", category)
    print("Recommended Text: ", text)
    print("User's Interests: ", user['interests'])
    if category == user['interests'][0]:
        correct_predictions += 1

print("\nAccuracy: ", correct_predictions/len(users))


# ### System Evaluation

# To calculate the accuracy, precision, recall, F-Measure, and AU-ROC, we'll need to modify the code to make it suitable for a multi-label classification problem. Since the user interests are not limited to one category, we'll create binary classifiers for each category and then calculate the mentioned metrics for each classifier.

# In[49]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Create binary classifiers for each category
categories = list(topics.keys())
category_classifiers = {category: (document_vectors, (balanced_data['Category'] == category).astype(int)) for category in categories}

# Calculate metrics for each user
all_true_labels = []
all_predicted_labels = []

for user in users:
    true_labels = [1 if interest in user['interests'] else 0 for interest in categories]
    predicted_labels = []

    for category in categories:
        classifier_data, category_labels = category_classifiers[category]
        best_similarity = -1
        match_index = -1

        for j in range(0, classifier_data.shape[0]):
            document = classifier_data[j]
            similarity = cosine_similarity(document, user_vectors[user['id'] - 1])

            if similarity > best_similarity:
                best_similarity = similarity
                match_index = j

        predicted_label = category_labels.iloc[match_index]
        predicted_labels.append(predicted_label)

    all_true_labels.append(true_labels)
    all_predicted_labels.append(predicted_labels)

all_true_labels = np.array(all_true_labels)
all_predicted_labels = np.array(all_predicted_labels)

# Calculate accuracy, precision, recall, F-Measure, and AU-ROC
accuracy = accuracy_score(all_true_labels, all_predicted_labels)
precision = precision_score(all_true_labels, all_predicted_labels, average='micro')
recall = recall_score(all_true_labels, all_predicted_labels, average='micro')
f_measure = f1_score(all_true_labels, all_predicted_labels, average='micro')
au_roc = roc_auc_score(all_true_labels, all_predicted_labels, average='micro')

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F-Measure: ", f_measure)
print("AU-ROC: ", au_roc)


# In[50]:


from sklearn.metrics import classification_report, multilabel_confusion_matrix

# Calculate classification report and confusion matrix
class_report = classification_report(all_true_labels, all_predicted_labels, target_names=categories, zero_division=0)
conf_matrix = multilabel_confusion_matrix(all_true_labels, all_predicted_labels)

print("Classification Report:\n", class_report)
print("Confusion Matrix:\n")

for i, category in enumerate(categories):
    print(f"{category}:\n")
    print(conf_matrix[i])
    print()


# In[51]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Function to plot ROC curve for each category
def plot_roc_curve(true_labels, predicted_labels, category, ax):
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr,
             lw=2, label=f'{category} (area = {roc_auc:.2f})')
    
# Set up the figure and axis for the plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve for all categories')

# Plot ROC curve for each category
for category in categories:
    true_labels = all_true_labels[:, categories.index(category)]
    predicted_labels = all_predicted_labels[:, categories.index(category)]
    plot_roc_curve(true_labels, predicted_labels, category, ax)

ax.legend(loc="lower right")
plt.show()


# In[52]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Function to plot ROC curve for each category
def plot_roc_curve(true_labels, predicted_labels, category):
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for ' + category)
    plt.legend(loc="lower right")
    plt.show()

# Plot ROC curve for each category
for category in categories:
    true_labels = all_true_labels[:, categories.index(category)]
    predicted_labels = all_predicted_labels[:, categories.index(category)]
    plot_roc_curve(true_labels, predicted_labels, category)

