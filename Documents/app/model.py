from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import pymysql as mdb
import pandas as pd
import random
import numpy as np
from sklearn.externals import joblib
#from sklearn import tree
from sklearn import ensemble
from sklearn.metrics.pairwise import linear_kernel
from sklearn.linear_model import LinearRegression
import datetime
from operator import itemgetter
import scipy.special as sp
from app import app

def return_percentage(votes, subreddit):
    # make simple linear model for development of gaussian rank distribution with number of votes    
    
    #parameters: cutoff, slope 1, intercept 1, slope 2 intercept 2
    cent_params = {1:[1, -56, 70, -26, 40], 2:[1.1, -31.82, 37, 0, 2], 3:[1.6, -10, 15, 0, 5]}    
    width_params = {1:[1, -10, 15, 0, 5], 2:[1.1, 0, 7, 0, 7], 3:[1.6, 0, 5, 0, 5]}
    
    logvotes = np.log(1 + np.log10(votes + 1))

    cp = cent_params[subreddit]
    wp = width_params[subreddit]

    if logvotes < cp[0]:
        center = logvotes*cp[1] + cp[2]
        hwhm = logvotes*wp[1] + wp[2]
    else:
        center = logvotes*cp[3] + cp[4]
        hwhm = logvotes*wp[3] + wp[4]
        
    gauss_width = 2*hwhm/(2 * np.sqrt(2*np.log(2)))

    x1 = (25 - center)/(2 * gauss_width)
    
    integral = (sp.erf(x1) + 1)        
    percentage = integral/2
    
    return round(100*percentage,1), int(round(center)), hwhm

def find_similar_posts(X_test, X_ref, number):
    
    cos_similarities = linear_kernel(X_test, X_ref)    
    related_texts = cos_similarities.flatten().argsort()[-1:(-1-number):-1]
    
    return related_texts, cos_similarities[:,related_texts]

def find_keywords(test_text, X_text, Yups, vectorizer,  number):
    
    title = vectorizer.transform([test_text])
    vector = np.zeros((1, X_text[0,:].shape[1]))

    X_in = X_text.toarray()  

    for ind, val in enumerate(Yups):
        vector = vector + X_in[ind, :]*val
       
    for ind, val in enumerate(title.toarray()):
        vector[ind] = vector[ind]*val
        
    best_inds = vector.flatten().argsort()[-1:(-1-number):-1]
    
    return itemgetter(*best_inds)(vectorizer.get_feature_names())
    

def make_lists(data_set):
    
    vocablist = []
    txtlist = []
    titlelist = []
    for post in data_set.iterrows():
        fname = 'text/' + post[1]['id']
        txtread = open(fname, 'r')
        tstr = unicode(txtread.read(), errors='ignore')
        txtread.close
        txtlist.append(tstr)
      
        titlelist.append(unicode(post[1]['title'], errors = 'ignore'))
    #        vocablist.append(tstr + ' ' + unicode(post[1]['title'], errors = 'ignore'))  
        vocablist.append(unicode(post[1]['title'], errors = 'ignore'))   
    
    print 'training lists made'
    
    return vocablist, titlelist, txtlist

def make_vectorizer(train_set, max_feat, vocab):
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
        vocablist.append(unicode(post[1]['title'], errors = 'ignore'))  
    
    print 'training lists made'
    
    cutoff = train_set.shape[0]
    
    # initialize tfidf vectorizer
    #txt_vect = TfidfVectorizer(ngram_range = (1,1), analyzer='word')
    title_vect = TfidfVectorizer(ngram_range = (1,2), analyzer='word', stop_words = 'english', max_features = max_feat, vocabulary = vocab)
    # calculate term-document matrix
    #X_txt = txt_vect.fit_transform(txtlist)

    title_vect.fit(vocablist)
    
    X_title = title_vect.transform(titlelist)
    X_text = title_vect.transform(txtlist)
    
    return title_vect, X_title, X_text

def train_choose_sub(train_set, Y, vocab = None):
        
    title_vect, X_title, X_text = make_vectorizer(train_set.loc[:, ['id', 'title']], 10000, vocab)

    # Make compound X vector from title and text vectors
    X = np.concatenate((X_title.toarray(), X_text.toarray()), axis = 1)
    
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
    
    return theta.predict(Xp), Xp

def train_linear(X_title, X_text, Y):

    # Make compound X vector from title and text vectors
    X = np.concatenate((X_title.toarray(), X_text.toarray()), axis = 1)
    
    # Train naive bayes
    theta = LinearRegression()
    theta.fit(X, Y)

    print 'model trained'    
    
    return theta
    
def test_linear(theta, Xp_title, Xp_text):
    
    Xp = np.concatenate((Xp_title.toarray(), Xp_text.toarray()), axis = 1)
    
    return theta.predict(Xp)
    
def train_ups(train_set, X_title, X_text, Y, depth, num_leaf, feature_list, num_posts, ftype):
    
    #step through training set, find 5 most similar articles, and create X vector from those posts
    
    Xd = []
    
    #feature_list = ['Yups','txt_sim', 'title_sim', 'txt_sent', 'title_sent', 'age', 'num_chars', 'create_dow', 'create_tod'];   
      
    test_list = [train_set.loc[:,'created_utc'].values, train_set.loc[:,'num_chars'].values, train_set.loc[:,'txt_sent'].values]        
      
    title_similarities = linear_kernel(X_title, X_title)
    text_similarities = linear_kernel(X_text, X_text)
    
    for ind in range(X_title.shape[0]):
    
        related_texts = text_similarities[ind,:].flatten().argsort()[-2:(-2-num_posts):-1]
            
        post_time = test_list[0][ind]          
            
        sim_posts = train_set.iloc[related_texts, :]
        sim_posts['txt_sim'] = text_similarities[ind, related_texts]
        sim_posts['title_sim'] = title_similarities[ind, related_texts]
        sim_posts['age'] = (post_time - sim_posts['created_utc'])/3600
        
        # initialize with tod post created
        
        tod = int(datetime.datetime.fromtimestamp(int(post_time)).strftime("%H")) 
        
        sim_posts = sim_posts.sort('Yups', ascending = False)        
        
        Xs = [tod, test_list[1][ind], test_list[2][ind]]
        for post in sim_posts.iterrows():
            Xs = Xs + post[1].loc[feature_list].values.tolist() 
        
        Xd.append(Xs)
        
    Xd = np.array(Xd)
    
    #tree_regressor = tree.DecisionTreeRegressor(max_depth = depth, min_samples_leaf = num_leaf)
    #tree_regressor = ensemble.RandomForestRegressor(n_estimators = 20, max_depth = depth, min_samples_leaf = num_leaf, bootstrap = False)
    if ftype == "RF":
        tree_regressor = ensemble.RandomForestRegressor(n_estimators = depth)
    else:
        tree_regressor = ensemble.GradientBoostingRegressor(n_estimators = depth, loss = 'quantile')
    
    dec_tree = tree_regressor.fit(Xd, Y)    

    print 'model trained'    
    
    return dec_tree
    
def test_ups(theta, test_list, train_set, Xp_title, Xp_text, Xt_title, Xt_text, feature_list, num_posts):
    Xd = []    
    
    post_time = test_list[0]    
    
    title_similarities = linear_kernel(Xt_title, Xt_title)
    text_similarities = linear_kernel(Xp_text, Xt_text)
    
    for ind in range(Xp_title.shape[0]):
    
        related_texts = text_similarities[ind,:].flatten().argsort()[-2:(-2-num_posts):-1]
            
        sim_posts = train_set.iloc[related_texts, :]
        sim_posts['txt_sim'] = text_similarities[ind, related_texts]
        sim_posts['title_sim'] = title_similarities[ind, related_texts]
        sim_posts['age'] = (post_time[ind] - sim_posts['created_utc'])/3600
        
        tod = int(datetime.datetime.fromtimestamp(int(post_time[ind])).strftime("%H"))     
        
        sim_posts = sim_posts.sort('Yups', ascending = False)                   
        
        Xs = [tod, test_list[1][ind], test_list[2][ind]]
        for post in sim_posts.iterrows():
            Xs = Xs + post[1].loc[feature_list].values.tolist() 
         
        Xd.append(Xs)
        
    Xd = np.array(Xd)

    return theta.predict(Xd)
    
def test_class():
    #test subreddit classification

    current_app = app.config.from_pyfile("aws.cfg")

    con = mdb.connect(user=current_app.config['DB_USER'], passwd=current_app.config['DB_PASS'], host=current_app.config['DB_HOST'], db=current_app.config['DB_NAME'])   

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
    
    Y_pred = test_sub(vectorizer, theta, test_set)
    
    cm = confusion_matrix(Y_te, Y_pred)

    con.close()
    
    return cm
    
