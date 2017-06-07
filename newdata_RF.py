import pandas as pd
import csv,re,sys,spacy
from pymongo import MongoClient
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from string import punctuation,printable
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from bs4 import BeautifulSoup
import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


'''
Save the new data as a .json file.
'''


def run_one(json_filename='one.json', rf_filename='RFClassifier.pkl'):
    corpus = soup(filename = json_filename)
    nlp = spacy.load('en')
    STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m"] + list(ENGLISH_STOP_WORDS))
    #tokenize all descriptions
    token_docs = [tokenize_string(doc) for doc in corpus]
    print("tokenized")
    #lemmatize
    lemmatized = [lemmatize_string(doc) for doc in token_docs]
    print("lemmatized")
    #remove stopwords
    no_stop = [remove_stopwords(doc, STOPLIST) for doc in lemmatized]
    print("stop words removed")
    tfidf_vectorized = tfideffed(no_stop)
    print('new_data_vectorized')
    #read in pickled model
    rf = un_pickle(rf_filename)
    #return probability of being fraud
    prob = rf.predict_proba(tfidf_vectorized)
    return prob[:,1]



def load_data(filename):
    df = pd.read_json(filename)
    descriptions = df["description"].tolist()
    return descriptions

def soup(filename = 'outfile.json'):
    '''
    Input: filename
    Return:
    '''
    descrip = []
    with open(filename) as json_data:
        file_json = json.load(json_data)
    for item in file_json:
        descrip.append(BeautifulSoup(item['description'], 'lxml').text)
    return descrip

def tokenize_string(doc):
    # First remove punctuation form string
    # PUNCT_DICT = {ord(punc): None for punc in punctuation}
    # doc = doc.translate(PUNCT_DICT)
    # remove junk
    clean_doc = ""
    for char in doc:
        if char in printable:
            clean_doc += char
    # Run the doc through spaCy
    nlp = spacy.load('en')
    tokenized_doc = nlp(clean_doc)
    return tokenized_doc

def lemmatize_string(doc):
    # Lemmatize and lower text
    lem_doc = " ".join([re.sub("\W+","",token.lemma_.lower()) if token.lemma_ != "-PRON-" else token.lower_ for token in doc])
    return lem_doc

def remove_stopwords(doc, stop_words):
    no_stop = " ".join([token for token in doc.split() if token not in stop_words])
    return no_stop

def count_vec(processed):
    countvect = CountVectorizer()
    count_vectorized = countvect.fit_transform(processed)
    return count_vectorized
    #vocab = countvect.get_feature_names()


def pickle_tfidf(trained_processed):
    # this was only needed once in order to pickle the tfidf_vectorizer so that we can transform the new data
    tfidfvect = TfidfVectorizer()
    tfidf_vectorized = tfidfvect.fit_transform(trained_processed)
    to_pickle('data/tfidf_vectorizer.pkl', tfidfvect)

def tfideffed(processed):
    #tfidfvect = TfidfVectorizer()
    #tfidf_vectorized = tfidfvect.fit_transform(processed)
    tfidfvect =un_pickle('data/tfidf_vectorizer.pkl')
    new_tfidf = tfidfvect.transform(processed)
    return new_tfidf

def un_pickle(filename):
    with open(filename, 'rb') as fp:
        processed = pickle.load(fp)
    return processed

def to_pickle(filename, object_tp):
    with open(filename, 'wb') as fp:
        pickle.dump(object_tp,fp)

if __name__ == '__main__':
    #only needed to call these next two lines once
    #all_processed = un_pickle('data/all_processed.pkl')
    #pickle_tfidf(all_processed)

    new_prob = run_one(json_filename = 'data/data_point.json', rf_filename='data/RFCModel.pkl')

    print(new_prob)
