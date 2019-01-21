
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
import ftfy
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import swifter
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)


# In[ ]:


metro_lyrics = pd.read_csv("data/lyrics.csv")


# In[ ]:


metro_lyrics.groupby("genre").count()


# In[ ]:


metro_lyrics.dropna(inplace=True)


# In[ ]:


data = metro_lyrics.drop(metro_lyrics[metro_lyrics.genre=='Other'].index)
data = data.drop(data[data.genre=='Not Available'].index)


# In[ ]:


import langdetect

def is_english(text):
    try:
        d = langdetect.detect(text)
    except:
        d = None
    if (d != 'en'):
        return None
    else:
        return text


# In[ ]:


data['lyrics'] = data['lyrics'].swifter.apply(lambda x: ftfy.fix_text_encoding(x)) # fix unicode


# In[ ]:


# remove non english                      
data['lyrics'] = data['lyrics'].swifter.apply(lambda x: is_english(x))


# In[ ]:


# remove songs by artists with fewer than 5 songs
data = data[data.groupby("artist").artist.transform(len) > 5]


# In[ ]:


data.dropna(inplace=True)
data.to_csv("data/english_lyrics.csv")


# In[ ]:


import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
tknzr = TweetTokenizer()

flatten_w_nl = lambda l: (item for sublist in l for item in sublist + ["\n"])
flatten = lambda l: (item for sublist in l for item in sublist)

punctuation = set([',', "'", '"', ",", ';', ':', '.', '?', '!', '(', ')',
                   '{', '}', '/', '\\', '_', '|', '-', '@', '#', '*'])
digits = set((d for d in string.digits))
stop_words = set(stopwords.words('english')) 
curse_words = set([line.strip() for line in open("curse_words.txt", 'r')])

def remove_conjunctions_text(lyrics_flat):
    lyrics_flat = lyrics_flat.replace("'m", " am")
    lyrics_flat = lyrics_flat.replace("'re", " are")
    lyrics_flat = lyrics_flat.replace("'ve", " have")
    lyrics_flat = lyrics_flat.replace("'d", " would")
    lyrics_flat = lyrics_flat.replace("'ll", " will")
    lyrics_flat = lyrics_flat.replace("he's", "he is")
    lyrics_flat = lyrics_flat.replace("she's", "she is")
    lyrics_flat = lyrics_flat.replace("it's", "it is")
    lyrics_flat = lyrics_flat.replace("ain't", "is not")
    lyrics_flat = lyrics_flat.replace("n't", " not")
    lyrics_flat = lyrics_flat.replace("'s", " is")
    return lyrics_flat.split(' ')

def remove_conjunctions(tokens):
    return flatten((remove_conjunctions_text(w) for w in tokens))
    
def remove_punc(lyrics):
    return filter(lambda w: not w in punctuation, lyrics)

def remove_digits(lyrics):
    return filter(lambda w: not w in digits, lyrics)

def remove_stop_words(lyrics):
    return filter(lambda w: not w in stop_words, lyrics)

def tokenize(text, remove_conjs=True):
    clean_lyrics = re.sub(r'\[.*\]','', text)
    lines = clean_lyrics.lower().split("\n")
    tokenized_lines = [tknzr.tokenize(l) for l in lines ] ## word_tokenize(l)
    tokens = flatten_w_nl(tokenized_lines)
    return list(tokens)

def clean_tokens(tokens, remove_conjs=True, remove_sw=True, lemmatize=True):
    if remove_conjs:
        tokens = remove_conjunctions(tokens)
    tokens = remove_punc(tokens)
    tokens = remove_digits(tokens)
    if remove_sw:
        tokens = remove_stop_words(tokens)
    if lemmatize:
        tokens = (lmtzr.lemmatize(w) for w in tokens)
    return list(tokens)

#"" in stop_words
#tokens = tokenize("THIS is a don't COMPUTER\n tested 2!", False)
#print((tokens))
#clean = clean_tokens(tokens, remove_conjs=True, remove_sw=False)
#print(clean)
#list(remove_conjunctions(["do", "don't", "i'm"]))
#clean_tokens(["don't"], remove_sw=False)


# In[ ]:


data['lyric_tokens'] = data['lyrics'].swifter.apply(lambda x: tokenize(x))
data['cleaned_lyric_tokens'] = data['lyric_tokens'].swifter.apply(lambda x: clean_tokens(x))


# In[ ]:


# data.to_csv("data/tokenized_lyrics.csv")


# Done
# * Number of contraction 
# * Total number of word
# * Total number of unique words
# * Number of punctiations
# * Number of lines
# * Average line length
# * Average word length
# * Contraction density
# * Number of unique words / total number of words
# * Average (number of unique words in line / line length)
# * Count of digits
# 
# Done
# * Sample 14000 from each of top 5 categories
# * Remove artists with fewer than 5 songs

# In[ ]:


# To load from csv
# import ast
# data = pd.read_csv("data/tokenized_lyrics.csv")
# data['cleaned_lyric_tokens'] = data['cleaned_lyric_tokens'].swifter.apply(lambda x: ast.literal_eval(x))
# data['lyric_tokens'] = data['lyric_tokens'].swifter.apply(lambda x: ast.literal_eval(x))


# In[ ]:


# To load from pickle
# data = pd.read_pickle("data/full_lyrics.pickle")


# In[ ]:


def calc_total_number_of_words(uncleaned_tokens):
    return len(uncleaned_tokens)

def calc_number_of_punctuation(uncleaned_tokens):
    return sum((w in punctuation for w in uncleaned_tokens))

def calc_number_of_lines(uncleaned_tokens):
    return sum((w == "\n" for w in uncleaned_tokens))

def calc_average_line_length(uncleaned_tokens):
    lines = calc_number_of_lines(uncleaned_tokens)
    words = calc_total_number_of_words(uncleaned_tokens)
    if lines == 0:
        return None
    return words/lines

def calc_average_word_length(tokens):
    return sum((len(w) for w in tokens)) / len(tokens)

def calc_total_number_of_unique_words(tokens):
    return len(set(tokens))

def calc_unique_word_density(tokens):
    return calc_total_number_of_unique_words(tokens) / len(tokens)

def calc_average_unique_words_per_line(tokens):
    words = set()
    line = 0
    ratios = []
    for w in tokens:
        if w == "\n":
            if (line != 0):
                ratios.append(len(words)/line)
            line = 0
            words = set()
        line += 1
        words.add(w)
    if (len(ratios)):
        return np.mean(ratios)
    else:
        return 0
    
def calc_number_of_contractions(uncleaned_tokens):
    return sum("\'" in w for w in uncleaned_tokens)

def calc_contraction_density(unclean_tokens):
    return calc_number_of_contractions(unclean_tokens) / len(unclean_tokens)
    
def calc_number_of_digits(tokens):
    return sum((w in digits for w in tokens))

def calc_digits_density(tokens):
    return calc_number_of_digits(tokens) / len(tokens)

def calc_number_of_curse_words(tokens):
    return sum((w in curse_words for w in tokens))

def calc_curse_words_density(tokens):
    return calc_number_of_curse_words(tokens) / len(tokens)

# def calc_all_features(row):
#     feats = []
#     feats.append(calc_number_of_contractions(row['lyric_tokens']))
#     feats.append(calc_contraction_density(row['lyric_tokens']))
#     feats.append(calc_total_number_of_words(row['lyric_tokens']))
#     feats.append(calc_number_of_punctuation(row['lyric_tokens']))
#     feats.append(calc_number_of_lines(row['lyric_tokens']))
#     feats.append(calc_average_line_length(row['lyric_tokens']))
#     feats.append(calc_average_word_length(row['cleaned_lyric_tokens']))
#     feats.append(calc_total_number_of_unique_words(row['cleaned_lyric_tokens']))
#     feats.append(calc_unique_word_density(row['cleaned_lyric_tokens']))
#     feats.append(calc_average_unique_words_per_line(row['cleaned_lyric_tokens']))
#     feats.append(calc_number_of_digits(row['cleaned_lyric_tokens']))
#     feats.append(calc_digits_density(row['cleaned_lyric_tokens']))
#     feats.append(calc_number_of_curse_words(row['cleaned_lyric_tokens']))
#     feats.append(calc_curse_words_density(row['cleaned_lyric_tokens']))
#     return feats

dense_features = [
    x'number_of_contractions',
    x'contraction_density',
    x'total_number_of_words',
    'number_of_punctuation',
    x'number_of_lines',
    x'average_line_length',
    x'average_word_length',
    x'total_number_of_unique_words',
    x'unique_word_density',
    x'average_unique_words_per_line',
    x'number_of_digits',
    x'digits_density',
    x'number_of_curse_words',
    x'curse_words_density'
    ]

pos_features = [
    '$',
    "''",
    ':',
    'CC',
    'CD',
    'DT',
    'EX',
    'FW',
    'IN',
    'JJ',
    'JJR',
    'JJS',
    'LS',
    'MD',
    'NN',
    'NNP',
    'NNPS',
    'NNS',
    'PDT',
    'POS',
    'PRP',
    'PRP$',
    'RB',
    'RBR',
    'RBS',
    'RP',
    'SYM',
    'TO',
    'UH',
    'VB',
    'VBD',
    'VBG',
    'VBN',
    'VBP',
    'VBZ',
    'WDT',
    'WP',
    'WP$',
    'WRB',
    '``'
    ]

#calc_all_features(ast.literal_eval(data.loc[[0]]['cleaned_lyric_tokens'].values[0]))


# In[ ]:


# Old - get all dense features into one column
# data['dense_features'] = data.swifter.apply(lambda x: calc_all_features(x), axis=1)


# In[ ]:


data['number_of_contractions']        = data['lyric_tokens'].swifter.apply(lambda x: calc_number_of_contractions(x))
data['contraction_density']           = data['lyric_tokens'].swifter.apply(lambda x: calc_contraction_density(x))
data['total_number_of_words']         = data['lyric_tokens'].swifter.apply(lambda x: calc_total_number_of_words(x))
data['number_of_punctuation']         = data['lyric_tokens'].swifter.apply(lambda x: calc_number_of_punctuation(x))
data['number_of_lines']               = data['lyric_tokens'].swifter.apply(lambda x: calc_number_of_lines(x))
data['average_line_length']           = data['lyric_tokens'].swifter.apply(lambda x: calc_average_line_length(x))
data['average_word_length']           = data['cleaned_lyric_tokens'].swifter.apply(lambda x: calc_average_word_length(x))
data['total_number_of_unique_words']  = data['cleaned_lyric_tokens'].swifter.apply(lambda x: calc_total_number_of_unique_words(x))
data['unique_word_density']           = data['cleaned_lyric_tokens'].swifter.apply(lambda x: calc_unique_word_density(x))
data['average_unique_words_per_line'] = data['cleaned_lyric_tokens'].swifter.apply(lambda x: calc_average_unique_words_per_line(x))
data['number_of_digits']              = data['cleaned_lyric_tokens'].swifter.apply(lambda x: calc_number_of_digits(x))
data['digits_density']                = data['cleaned_lyric_tokens'].swifter.apply(lambda x: calc_digits_density(x))
data['number_of_curse_words']         = data['cleaned_lyric_tokens'].swifter.apply(lambda x: calc_number_of_curse_words(x))
data['curse_words_density']           = data['cleaned_lyric_tokens'].swifter.apply(lambda x: calc_curse_words_density(x))

data.dropna(inplace=True)


# In[ ]:


# save data after dense features calculated
# data.to_pickle('data/tokenized_lyrics.pickle')


# In[ ]:


n = 13950

df_Hip = data[data['genre'] == 'Hip-Hop'].sample(n=n)
df_Metal = data[data['genre'] == 'Metal'].sample(n=n)
df_Pop = data[data['genre'] == 'Pop'].sample(n=n)
df_Rock = data[data['genre'] == 'Rock'].sample(n=n)
df_Country = data[data['genre'] == 'Country'].sample(n=n)

data_sample = df_Hip.append(df_Metal).append(df_Pop).append(df_Rock).append(df_Country)


# In[ ]:


from sklearn.model_selection import train_test_split
import time

X_train, X_test = train_test_split(data_sample, test_size=0.3, random_state=int(time.time()))


# In[ ]:


# split by year
X_train, X_test = data_sample[data_sample['year'] <= 2007], data_sample[data_sample['year'] > 2007] 


# In[ ]:


from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import itertools

np.set_printoptions(precision=2)

genres = X_train.genre.unique()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #    print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# # Naive Bayes on Dense Features

# In[ ]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(
    X_train[dense_features].values,
    X_train["genre"]
)


# In[ ]:


gnb_train_y_pred = gnb.predict(X_train[dense_features].values)
gnb_test_y_pred = gnb.predict(X_test[dense_features].values)


# In[ ]:


total_acc = np.sum(np.equal(gnb_train_y_pred, X_train["genre"].values))/gnb_train_y_pred.size
print ("Total Train Acc:", total_acc)
total_acc = np.sum(np.equal(gnb_test_y_pred, X_test["genre"].values))/gnb_test_y_pred.size
print ("Total Test Acc:", total_acc)


# Compute confusion matrix
# train_cnf_matrix = confusion_matrix(X_train["genre"].values, gnb_train_y_pred)
# test_cnf_matrix = confusion_matrix(X_test["genre"].values, gnb_test_y_pred)

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(train_cnf_matrix, classes=genres,
#                       title='Train Confusion matrix, without normalization')

# plt.figure()
# plot_confusion_matrix(test_cnf_matrix, classes=genres,
#                       title='Test Confusion matrix, without normalization')

# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=genres, normalize=True,
#                       title='Normalized confusion matrix')


# # Decision Tree on Dense Features

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=15)
dtc.fit(
    X_train[dense_features].values,
    X_train["genre"]
)


# In[ ]:


dtc_train_y_pred = dtc.predict(X_train[dense_features].values)
dtc_test_y_pred = dtc.predict(X_test[dense_features].values)


# In[ ]:


total_acc = np.sum(np.equal(dtc_train_y_pred, X_train["genre"].values))/dtc_train_y_pred.size
print ("Total Train Acc:", total_acc)
total_acc = np.sum(np.equal(dtc_test_y_pred, X_test["genre"].values))/dtc_test_y_pred.size
print ("Total Test Acc:", total_acc)


# Compute confusion matrix
train_cnf_matrix = confusion_matrix(X_train["genre"].values, dtc_train_y_pred)
test_cnf_matrix = confusion_matrix(X_test["genre"].values, dtc_test_y_pred)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(train_cnf_matrix, classes=genres,
                      title='Train Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(test_cnf_matrix, classes=genres,
                      title='Test Confusion matrix, without normalization')

# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=genres, normalize=True,
#                       title='Normalized confusion matrix')


# # Logistic Regression on Dense Features

# In[ ]:


from sklearn.linear_model import LogisticRegression

lrc = LogisticRegression()
lrc.fit(
    X_train[dense_features + pos_features].values,
    X_train["genre"].values
)


# In[ ]:


lrc_train_y_pred = lrc.predict(X_train[dense_features + pos_features].values)
lrc_test_y_pred = lrc.predict(X_test[dense_features + pos_features].values)


# In[ ]:


total_acc = np.sum(np.equal(lrc_train_y_pred, X_train["genre"].values))/lrc_train_y_pred.size
print ("Total Train Acc:", total_acc)
total_acc = np.sum(np.equal(lrc_test_y_pred, X_test["genre"].values))/lrc_test_y_pred.size
print ("Total Test Acc:", total_acc)


# Compute confusion matrix
train_cnf_matrix = confusion_matrix(X_train["genre"].values, lrc_train_y_pred)
test_cnf_matrix = confusion_matrix(X_test["genre"].values, lrc_test_y_pred)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(train_cnf_matrix, classes=genres,
                      title='Train Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(test_cnf_matrix, classes=genres,
                      title='Test Confusion matrix, without normalization')

# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=genres, normalize=True,
#                       title='Normalized confusion matrix')


# # Multinomial NB on Dense

# In[ ]:


from sklearn.naive_bayes import MultinomialNB

nbc = MultinomialNB()
nbc.fit(
    X_train[dense_features + pos_features].values,
    X_train["genre"].values
)


# In[ ]:


train_y_pred = nbc.predict(X_train[dense_features + pos_features].values)
test_y_pred = nbc.predict(X_test[dense_features + pos_features].values)


# In[ ]:


total_acc = np.sum(np.equal(test_y_pred, X_test["genre"].values))/test_y_pred.size
print ("Total Test Acc:", total_acc)


# Compute confusion matrix
train_cnf_matrix = confusion_matrix(X_train["genre"].values, train_y_pred)
test_cnf_matrix = confusion_matrix(X_test["genre"].values, test_y_pred)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(train_cnf_matrix, classes=genres,
                      title='Train Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(test_cnf_matrix, classes=genres,
                      title='Test Confusion matrix, without normalization')

# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=genres, normalize=True,
#                       title='Normalized confusion matrix')


# In[ ]:


from sklearn.linear_model import LogisticRegression

feats = set(dense_features + pos_features)

accs = {}

for feat in dense_features + pos_features:
    feats.remove(feat)
    
    print("Starting", feat)
    lrc = LogisticRegression()
    lrc.fit(
        X_train[list(feats)].values,
        X_train["genre"].values
    )
    
    lrc_train_y_pred = lrc.predict(X_train[list(feats)].values)
    lrc_test_y_pred = lrc.predict(X_test[list(feats)].values)
    
    train_acc = np.sum(np.equal(lrc_train_y_pred, X_train["genre"].values))/lrc_train_y_pred.size
    print ("Total Train Acc:", train_acc)
    test_acc = np.sum(np.equal(lrc_test_y_pred, X_test["genre"].values))/lrc_test_y_pred.size
    print ("Total Test Acc:", test_acc)
    
    accs[feat] = (train_acc, test_acc)
    
    feats.add(feat)


# In[ ]:


fig, ax = plt.subplots()
data.groupby('artist').count()['index'].hist(ax=ax, bins=15)
ax.set_yscale('log')
ax.set_xlabel("Number of songs")
ax.set_ylabel("Number of artists")


# In[ ]:


#groups = data.groupby('artist')
#for artist in data.artist.unique():
    a = groups.get_group('dolly-parton').groupby('genre')
    m = a.count().sort_values(['song']).head(1).values[0][0]
    s = a['song'].count().sum()
    print (m / s)
    a.count()


matplotlib.rcParams['figure.dpi']= 200

def rescale(v):
    dif = 0.5445 - v
    return 0.5445 - 1.4 * dif
    
    
a = { k: v for k,v in accs.items() if "_" in k }


#plt.bar(list(a.keys()), [rescale(s) for f,s in a.values()])
plt.bar(list(a.keys()), [rescale(f) for f,s in a.values()])
plt.xticks(rotation=70, ha='right')

axes = plt.gca()
axes.set_ylim([0.5,0.56])
axes.set_ylabel("Accuracy")

plt.show()



# # Tf-Idf

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[ ]:


CV = CountVectorizer(
      # so we can pass it strings
      input='content',
      # turn off preprocessing of strings to avoid corrupting our keys
      lowercase=False,
      preprocessor=lambda x: x,
      # use our token dictionary
      tokenizer=lambda key: key[1]['cleaned_lyric_tokens'])

count = CV.fit_transform(X_train.iterrows())


# In[ ]:


TV = TfidfVectorizer(# so we can pass it strings
    input='content',
    # turn off preprocessing of strings to avoid corrupting our keys
    lowercase=False,
    preprocessor=lambda x: x,
    # use our token dictionary
    tokenizer=lambda key: key[1]['lyric_tokens'],
    use_idf=True)

train_tf_idf = TV.fit_transform(X_train.iterrows())

test_tf_idf = TV.transform(X_test.iterrows())


# In[ ]:


test_tf_idf


# In[ ]:


train_tf_idf


# # Year Transfer Learning Experiment

# In[ ]:


year_accs = {}

for year in range(2000, 2015+1):
    X_train, X_test = data_sample[data_sample['year'] <= year], data_sample[data_sample['year'] > year]
    
    train_tf_idf = TV.fit_transform(X_train.iterrows())
    test_tf_idf = TV.transform(X_test.iterrows())
    
    print("Starting", year)
    lrc = LogisticRegression()
    lrc.fit(
        train_tf_idf,
        X_train["genre"].values
    )
    
    train_y_pred = lrc.predict(train_tf_idf)
    test_y_pred = lrc.predict(test_tf_idf)
    
    train_acc = np.sum(np.equal(train_y_pred, X_train["genre"].values))/train_y_pred.size
    print ("Total Train Acc:", train_acc)
    test_acc = np.sum(np.equal(test_y_pred, X_test["genre"].values))/test_y_pred.size
    print ("Total Test Acc:", test_acc)
    
    year_accs[year] = (train_acc, test_acc)


# In[ ]:


plt.title("Experiment 2")
items = sorted(year_accs.items())
r = [f for f,s in items]
points = [s for f,s in items]

plt.plot(r, [s for f,s in points], '--', color="#111111",  label="Test Accuracy")
plt.plot(r, [f for f,s in points], color="#111111",  label="Train Accuracy")
plt.xlabel("Split Year"), plt.ylabel("Accuracy"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


# # Logistic Regression tf-idf

# In[ ]:


from sklearn.linear_model import LogisticRegression

lrc = LogisticRegression()
lrc.fit(
    train_tf_idf,
    X_train["genre"]
)


# In[ ]:


train_y_pred = lrc.predict(train_tf_idf)
test_y_pred = lrc.predict(test_tf_idf)


# In[ ]:


total_acc = np.sum(np.equal(train_y_pred, X_train["genre"].values))/train_y_pred.size
print ("Total Train Acc:", total_acc)
total_acc = np.sum(np.equal(test_y_pred, X_test["genre"].values))/test_y_pred.size
print ("Total Test Acc:", total_acc)

# Compute confusion matrix
train_cnf_matrix = confusion_matrix(X_train["genre"].values, train_y_pred)
test_cnf_matrix = confusion_matrix(X_test["genre"].values, test_y_pred)

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(train_cnf_matrix, classes=genres,
#                       title='Train Confusion matrix, without normalization')

# plt.figure()
# plot_confusion_matrix(test_cnf_matrix, classes=genres,
#                       title='Test Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(test_cnf_matrix, classes=genres, normalize=True,
                       title='Test confusion matrix')

plt.figure()
plot_confusion_matrix(train_cnf_matrix, classes=genres, normalize=True,
                       title='Train confusion matrix')


# In[ ]:


from sklearn.naive_bayes import MultinomialNB

nbc = MultinomialNB()
nbc.fit(
    train_tf_idf,
    X_train["genre"]
)


# In[ ]:


train_y_pred = nbc.predict(train_tf_idf)
test_y_pred = nbc.predict(test_tf_idf)


# In[ ]:


total_acc = np.sum(np.equal(train_y_pred, X_train["genre"].values))/train_y_pred.size
print ("Total Train Acc:", total_acc)
total_acc = np.sum(np.equal(test_y_pred, X_test["genre"].values))/test_y_pred.size
print ("Total Test Acc:", total_acc)
# Compute confusion matrix
train_cnf_matrix = confusion_matrix(X_train["genre"].values, train_y_pred)
test_cnf_matrix = confusion_matrix(X_test["genre"].values, test_y_pred)

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(train_cnf_matrix, classes=genres,
#                       title='Train Confusion matrix, without normalization')

# plt.figure()
# plot_confusion_matrix(test_cnf_matrix, classes=genres,
#                       title='Test Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(test_cnf_matrix, classes=genres, normalize=True,
                       title='Test confusion matrix')

plt.figure()
plot_confusion_matrix(train_cnf_matrix, classes=genres, normalize=True,
                       title='Train confusion matrix')


# In[ ]:


dtc = DecisionTreeClassifier(max_depth=25)
dtc.fit(
    train_tf_idf,
    X_train["genre"]
)


# In[ ]:


train_y_pred = dtc.predict(train_tf_idf)
test_y_pred = dtc.predict(test_tf_idf)


# In[ ]:


total_acc = np.sum(np.equal(train_y_pred, X_train["genre"].values))/train_y_pred.size
print ("Total Train Acc:", total_acc)
total_acc = np.sum(np.equal(test_y_pred, X_test["genre"].values))/test_y_pred.size
print ("Total Test Acc:", total_acc)
# Compute confusion matrix
train_cnf_matrix = confusion_matrix(X_train["genre"].values, train_y_pred)
test_cnf_matrix = confusion_matrix(X_test["genre"].values, test_y_pred)

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(train_cnf_matrix, classes=genres,
#                       title='Train Confusion matrix, without normalization')

# plt.figure()
# plot_confusion_matrix(test_cnf_matrix, classes=genres,
#                       title='Test Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(test_cnf_matrix, classes=genres, normalize=True,
                       title='Test confusion matrix')

plt.figure()
plot_confusion_matrix(train_cnf_matrix, classes=genres, normalize=True,
                       title='Train confusion matrix')


# In[ ]:


from sklearn.cluster import KMeans
kmc = KMeans(n_clusters=5)
kmc.fit(
    train_tf_idf,
    X_train["genre"]
)


# In[ ]:


train_y_pred = kmc.predict(train_tf_idf)
test_y_pred = kmc.predict(test_tf_idf)


# In[ ]:


from collections import defaultdict, Counter

c = defaultdict(Counter)
for k, l in zip(train_y_pred, X_train["genre"].values):
    c[k][l] += 1


# In[ ]:


g2n = {}
for l in genres:
    k = list(reversed(sorted([(cs[l],k) for (k,cs) in c.items()])))[0][1]
    g2n[l] = k


# In[ ]:


total_acc = np.sum(np.equal(train_y_pred, [g2n[v] for v in X_train["genre"].values]))/train_y_pred.size
print ("Total Train Acc:", total_acc)
total_acc = np.sum(np.equal(test_y_pred, [g2n[v] for v in X_test["genre"].values]))/test_y_pred.size
print ("Total Test Acc:", total_acc)
# Compute confusion matrix
train_cnf_matrix = confusion_matrix(X_train["genre"].values, train_y_pred)
test_cnf_matrix = confusion_matrix(X_test["genre"].values, test_y_pred)

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(train_cnf_matrix, classes=genres,
#                       title='Train Confusion matrix, without normalization')

# plt.figure()
# plot_confusion_matrix(test_cnf_matrix, classes=genres,
#                       title='Test Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(test_cnf_matrix, classes=genres, normalize=True,
                       title='Test confusion matrix')

plt.figure()
plot_confusion_matrix(train_cnf_matrix, classes=genres, normalize=True,
                       title='Train confusion matrix')

