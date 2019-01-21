
# coding: utf-8

# In[1]:


from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import namedtuple, Counter
import time

import pandas as pd
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)


# In[14]:


data = pd.read_pickle("data/tokenized_lyrics.pickle")


# In[15]:


class TaggedDocGen(object):
    def __init__(self, df):
       self.df = df.sample(frac=1)
    
    def __len__(self):
        return self.df.shape[0]
        
    def __iter__(self):
        for index, row in self.df.iterrows():
            yield TaggedDocument(row['cleaned_lyric_tokens'], [index])


# In[30]:


n = 13950

df_Hip = data[data['genre'] == 'Hip-Hop'].sample(n=n)
df_Metal = data[data['genre'] == 'Metal'].sample(n=n)
df_Pop = data[data['genre'] == 'Pop'].sample(n=n)
df_Rock = data[data['genre'] == 'Rock'].sample(n=n)
df_Country = data[data['genre'] == 'Country'].sample(n=n)

data_sample = df_Hip.append(df_Metal).append(df_Pop).append(df_Rock).append(df_Country)


# In[31]:


from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(data_sample, test_size=0.3, random_state=int(time.time()))


# In[32]:


X_test.loc[[71948]]
X_train.shape

X_train.groupby('genre').count()


# In[33]:


import multiprocessing
cores = multiprocessing.cpu_count()

it = TaggedDocGen(X_train)

model = Doc2Vec(vector_size=250, window=10, negative=5, epochs=1, min_count=2, workers=cores, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)


# In[34]:


len(it)


# In[35]:


print("Training %s" % model)
model.train(it, total_examples=len(it), epochs=model.epochs)


# In[ ]:


#model.save("doc2vec2.model")


# In[ ]:


#model = Doc2Vec.load("doc2vec2.model")


# In[36]:


#lyrics["vec"] = lyrics.index.map(lambda x: list(model[x]))

model.docvecs#.offset2doctag#[71948]


# In[26]:


genres = data.genre.unique()


# In[27]:


X_train.shape, X_test.shape


# In[ ]:


train = np.array([np.array(l) for l in X_train["vec"].values])
test = np.array([np.array(l) for l in X_test["vec"].values])


# In[ ]:


from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(
    train,
    X_train["genre_num"]
)


# In[ ]:


y_pred = gnb.predict(test)


# In[ ]:


y_pred


# In[ ]:


np.sum(np.equal(y_pred, X_test["genre_num"].values))/y_pred.size


# In[ ]:


correct = Counter()
total = Counter()

for pred, truth in zip(y_pred, X_test["genre_num"].values):
    if pred == truth:
        correct[truth] += 1
    total[truth] += 1


# In[ ]:


for y in correct:
    print (n2g[y], ":", correct[y]/total[y])


# In[ ]:




