# Introduction

This repository contains the code for LDA model implementation with Python. The code has been written and tested by Oğuzhan Öztürk <oguzhanozturk0@outlook.com>. 

## Requirements

* numpy
* pandas
* gensim
* nltk
* spacy

## Data

The data used in this notebook is provided in CSV format. It can be loaded using pandas library.

``` python
data=pd.read_csv('CSV FILE PATH')
```

## Preprocessing

To prepare the text data for LDA, the following preprocessing steps have been performed:

1. Text cleaning- to remove special characters from the text
2. Tokenization - to convert the text into a list of words
3. Stopword Removal - to remove commonly occurring English words such as 'the', 'a', 'and', etc.
4. Bigram and Trigram Generation - to group together sequences of words that frequently appear together in the corpus.
5. Lemmatization - to convert words to their base or dictionary form.

## Model Building

The LDA model in Gensim has been used to build the topic model. The optimal number of topics has been identified using coherence score.

``` python
lda_model=gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=9,random_state=100,update_every=1,
                                         chunksize=100,passes=10,alpha='auto',per_word_topics=True)
```

## Visualization

PyLDAvis have been used to visualize the topics and their relationships. 

``` python
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
```
