from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import pymysql as mdb
import pandas as pd
import random
import numpy as np
from sklearn.externals import joblib


def train_choose_sub(train_set, Y):
        
    txtlist = []
    titlelist = []
    vocablist = []
    for post in train_set.iterrows():
        fname = 'text/' + post[1]['id']
        txtread = open(fname, 'r')
        tstr = unicode(txtread.read(), errors='ignore')
        txtread.close
        txtlist.append(tstr)
      
        titlelist.append(unicode(post[1]['title'], errors = 'ignore'))
        vocablist.append(tstr + ' ' + unicode(post[1]['title'], errors = 'ignore'))  
    
    print 'training lists made'
    
    cutoff = train_set.shape[0]
    
    # initialize tfidf vectorizer
    #txt_vect = TfidfVectorizer(ngram_range = (1,1), analyzer='word')
    title_vect = TfidfVectorizer(ngram_range = (1,1), analyzer='word', min_df = 2.01/cutoff, max_df = 0.02)
    # calculate term-document matrix
    #X_txt = txt_vect.fit_transform(txtlist)
    
    title_vect.fit(vocablist)
    
    a = title_vect.transform(['word'])
    a.shape
    
    X_title = title_vect.transform(titlelist)
    X_test = title_vect.transform(txtlist)
    
    vocablist = None
    
    print 'training vectors generated'
    
    # Make compound X vector from title and text vectors
    X = np.concatenate((X_title.toarray(), X_test.toarray()), axis = 1)
    
    X.shape
    
    # Train naive bayes
    gnb = GaussianNB()
    theta = gnb.fit(X, Y)

    print 'model trained'    
    
    return title_vect, theta

def test_subred(vectorizer, theta, test_set):
    ptxtlist = []
    ptitlelist = []

    for post in test_set.iterrows():
        fname = 'text/' + post[1]['id']
        txtread = open(fname, 'r')
        tstr = unicode(txtread.read(), errors='ignore')
        txtread.close
        ptxtlist.append(tstr)
      
        ptitlelist.append(unicode(post[1]['title'], errors = 'ignore'))
    
    print 'test lists made'

    X_p_title = vectorizer.transform(ptitlelist)
    X_p_test = vectorizer.transform(ptxtlist)
    Xp = np.concatenate((X_p_title.toarray(), X_p_test.toarray()), axis = 1)
    
    return theta.predict(Xp)

def train_ups(train_set, Y):
        
    txtlist = []
    titlelist = []
    vocablist = []
    for post in train_set.iterrows():
        fname = 'text/' + post[1]['id']
        txtread = open(fname, 'r')
        tstr = unicode(txtread.read(), errors='ignore')
        txtread.close
        txtlist.append(tstr)
      
        titlelist.append(unicode(post[1]['title'], errors = 'ignore'))
        vocablist.append(tstr + ' ' + unicode(post[1]['title'], errors = 'ignore'))  
    
    print 'training lists made'
    
    cutoff = train_set.shape[0]
    
    # initialize tfidf vectorizer
    #txt_vect = TfidfVectorizer(ngram_range = (1,1), analyzer='word')
    title_vect = TfidfVectorizer(ngram_range = (1,1), analyzer='word', min_df = 2.01/cutoff, max_df = 0.02)
    # calculate term-document matrix
    #X_txt = txt_vect.fit_transform(txtlist)
    
    title_vect.fit(vocablist)
    
    a = title_vect.transform(['word'])
    a.shape
    
    X_title = title_vect.transform(titlelist)
    X_test = title_vect.transform(txtlist)
    
    vocablist = None
    
    print 'training vectors generated'
    
    # Make compound X vector from title and text vectors
    X = np.concatenate((X_title.toarray(), X_test.toarray()), axis = 1)
    
    X.shape
    
    # Train naive bayes
    gnb = GaussianNB()
    theta = gnb.fit(X, Y)

    print 'model trained'    
    
    return title_vect, theta
    
def test_class():
    #test subreddit classification
    
    
    con = mdb.connect('localhost', 'pyconnect', 'pypa$$1', 'Reddit')    

    strSQL = "Select id, title, txt_sent, title_sent, num_chars, subred from PostData Where txt_dl = 1;"
    outdf = pd.io.sql.read_sql(sql = strSQL, con = con)
    
    inds = range(outdf.shape[0])
    
    random.shuffle(inds)
    
    cutoff = int(round(0.7*outdf.shape[0]))
    train_inds = inds[:cutoff]
    test_inds = inds[cutoff:]
    
    train_set = outdf.iloc[train_inds,:]
    test_set = outdf.iloc[test_inds,:]
    Y_tr = outdf.loc[train_inds,'subred']
    Y_te = outdf.loc[train_inds,'subred']
    
    vectorizer, theta = train_choose_sub(train_set, Y_tr)
    
    Y_pred = predict_subred(vectorizer, theta, test_set)
    
    cm = confusion_matrix(Y_te, Y_pred)
    
    return cm
    