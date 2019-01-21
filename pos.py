
# coding: utf-8

# In[2]:


import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
import swifter
nltk.download('averaged_perceptron_tagger')

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)


# In[ ]:


text = word_tokenize("And now for something completely different")
Counter((s for f,s in nltk.pos_tag(text)))


# In[ ]:


import os
os.chdir(r'C:\Users\Dell1\Desktop\CIS519')


# In[3]:


data = pd.read_pickle('data/tokenized_lyrics.pickle')


# In[4]:


data['cleaned_lyric_tokens'] = data['cleaned_lyric_tokens'].swifter.apply(lambda x: [w for w in x if len(w)])


# In[5]:


pos = data['cleaned_lyric_tokens'].swifter.apply(lambda x: Counter((s for f,s in nltk.pos_tag(x))))


# In[6]:


pos_df = pd.DataFrame(list(pos)).fillna(0)


# In[7]:


data_comb = pd.concat([data, pos_df], axis=1)


# In[8]:


data_comb.to_pickle('data/full_lyrics.pickle')


# In[10]:


list(data_comb.columns)


# In[ ]:




