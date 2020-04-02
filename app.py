# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import joblib

from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import re
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords

import xgboost as xgb

import nltk
nltk.download('stopwords')
SAFE_DIV = 0.0001
STOP_WORDS = stopwords.words("english")
import en_core_web_sm
nlp = en_core_web_sm.load()
word2tfidf=joblib.load('Artifacts/word2tfidf.pkl')

bst = xgb.Booster()  # init model
bst.load_model('Artifacts/model_ver.pkl')

# model=joblib.load('Artifacts/model_ver.pkl')
cols=joblib.load('Artifacts/columns.pkl')

from flask import Flask,render_template,request
app = Flask(__name__)

train_org= pd.read_csv('train.csv')

inference_point={}


q1="What is the name of first computer?"
q2="When was the first computer found?"

@app.route('/index')
def index():
    return render_template('index.html')

def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)


    porter = PorterStemmer()
    pattern = re.compile('\W')

    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)


    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x,features="lxml")
        x = example1.get_text()


    return x

def get_token_features(q1, q2):
    token_features = [0.0]*10

    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))

    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))

    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))


    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))

    #Average Token Length of both Questions
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features

def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return lcs_set

def tfidf_w2v(q):
    doc1 = nlp(q)
    # 384 is the number of dimensions of vectors
    mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)])
    for word1 in doc1:
        # word2vec
        vec1 = word1.vector
        # fetch df score
        try:
            idf = word2tfidf[str(word1)]
        except:
            idf = 0
        # compute final vec
        mean_vec1 += vec1 * idf
    mean_vec1 = mean_vec1.mean(axis=0)
    return mean_vec1

@app.route('/predict', methods=['POST'])
def predict():

        q1 = request.form['q1']
        q2 = request.form['q2']

        inference_point['freq_qid1']=train_org[train_org['question1']==q1].shape[0]
        inference_point['freq_qid2']=train_org[train_org['question2']==q2].shape[0]

        inference_point['q1len']=len(q1)
        inference_point['q2len']=len(q2)
        inference_point['q1_n_words']=len(q1.split(" "))
        inference_point['q2_n_words']=len(q2.split(" "))


        w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), q1.split(" ")))

        inference_point['word_Common'] =1.0 * len(w1 & w2)
        inference_point['word_Total']= 1.0 * (len(w1) + len(w2))
        inference_point['word_share']= 1.0 * len(w1 & w2)/(len(w1) + len(w2))
        inference_point['freq_q1+q2'] = inference_point['freq_qid1']+inference_point['freq_qid2']
        inference_point['freq_q1-q2'] = inference_point['freq_qid1']-inference_point['freq_qid2']



        q1=preprocess(q1)
        q2=preprocess(q2)

        token_features=get_token_features(q1,q2)

        inference_point["cwc_min"]      =  token_features[0]
        inference_point["cwc_max"]      =  token_features[1]
        inference_point["csc_min"]      =  token_features[2]
        inference_point["csc_max"]       = token_features[3]
        inference_point["ctc_min"]       = token_features[4]
        inference_point["ctc_max"]       = token_features[5]
        inference_point["last_word_eq"]  = token_features[6]
        inference_point["first_word_eq"] = token_features[7]
        inference_point["abs_len_diff"]  = token_features[8]
        inference_point["mean_len"]     =  token_features[9]

        inference_point['longest_substr_ratio']=len(list(lcs(q1,q2))[0])

        inference_point['token_set_ratio'] =fuzz.token_set_ratio(q1,q2)
        inference_point['token_sort_ratio'] =fuzz.token_sort_ratio(q1,q2)
        inference_point['fuzz_ratio'] =fuzz.QRatio(q1,q2)
        inference_point['fuzz_partial_ratio'] =fuzz.partial_ratio(q1,q2)




        q1_vec=tfidf_w2v(q1)
        q2_vec=tfidf_w2v(q2)

        for i in range(len(q1_vec)):
            inference_point[str(i)+'_x']=q1_vec[i]
            inference_point[str(i)+'_y']=q2_vec[i]

        inference_point['fuzz_ratio']
        len(inference_point)

# cols
        X=pd.DataFrame(inference_point,index=[0])
        X=X[cols]

        x = xgb.DMatrix(X)
        if(bst.predict(x)>0.5):
            return 'similar'
        else:
            return 'dissimilar'


@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
