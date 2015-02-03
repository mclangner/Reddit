from sklearn.externals import joblib
from bs4 import BeautifulSoup
import urllib2,cookielib
import numpy as np
import metrics
import datetime
import time
import model
import pandas as pd
import random

def samples(num_samples):
    sam = pd.read_excel("app/samples.xlsx")
    
    inds = random.sample(range(sam.shape[0]), num_samples)
    
    return sam.iloc[inds,:]
    

def split_soup(soup_txt):
    stops = ['. ', '! ', '? ', '?"', '!"', '."']
    replacements = ['\n', '\t', '\r', '<', '\'' ]
    st = soup_txt
    for rep in replacements:
        st = st.replace(rep, ' ')
    
    spin = [st]
    for stop in stops:
        spout = []
        for item in spin:
            spout = spout + item.split(stop)
        spin = spout
        
    return spout
    
def get_text_from_url(url):
#    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
#           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#           'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
#           'Accept-Encoding': 'none',
#           'Accept-Language': 'en-US,en;q=0.8',
#           'Connection': 'keep-alive'}
 
    urlspl = url.split('.')
    
    if urlspl[len(urlspl) - 1] == 'webm':
        status = 1;
        txt = 'webm - skipped'
    else:
    
        try:
            cj = cookielib.CookieJar()
            opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
            opener.addheaders = [('User-agent', 'Chrome/23.0.1271.64')]
                   
            page = opener.open(url)
            soup = BeautifulSoup(page.read())
            stxt = soup.get_text()
            
            sents = split_soup(stxt)
                    
            txt = ''
            for item in sents:
                str_item = item.strip()
                if len(str_item) > 0:
                    sratio = len(str_item.split())/float(len(str_item))
                    if sratio > 0.13:
                        txt = txt + ' ' + item.encode('latin-1', 'ignore')
            status = 0
        except urllib2.HTTPError:
            #txt = err.fp.read()
            txt = "HTTPError"
            status = 1
        except urllib2.URLError:
            txt = "URLError"
            status = 1
                
    return txt, status

def fit_url(title, url):
    
    url_txt, status = get_text_from_url(url)

    url_txt = unicode(url_txt, errors='ignore')    
    
    app_dir = "app/"  
    
    vectorizer = joblib.load(app_dir + 'Pickles/NBayes/nb_gen_vectorizer.pkl')
    theta = joblib.load(app_dir + 'Pickles/NBayes/nb_gen_theta.pkl') 
    
    Xp_title = vectorizer.transform([title])
    Xp_text = vectorizer.transform([url_txt])

    Xp = np.concatenate((Xp_title.toarray(), Xp_text.toarray()), axis = 1)
    
    subreddit = theta.predict(Xp)[0]

    if subreddit > 0:
        
        vectorizer = joblib.load(app_dir + 'Pickles/NBayes/nb_tech_vectorizer.pkl')
        theta = joblib.load(app_dir + 'Pickles/NBayes/nb_tech_theta.pkl') 
        
        Xp_title = vectorizer.transform([title])
        Xp_text = vectorizer.transform([url_txt])
    
        Xp = np.concatenate((Xp_title.toarray(), Xp_text.toarray()), axis = 1)
        
        subreddit = theta.predict(Xp)[0]
        
        base_dir = app_dir + 'Pickles/Forest/'            
        
        #Grab post data for relevant subreddit

        end_str = str(subreddit) + '.pkl'        
        
        train_set = pd.io.pickle.read_pickle(base_dir + 'train_set' + end_str)        
        
        title_vect = joblib.load(base_dir + 'vectorizer' + end_str)
        X_title = joblib.load(base_dir + 'X_title' + end_str)
        X_text = joblib.load(base_dir + 'X_text' + end_str)
        theta = joblib.load(base_dir + 'theta' + end_str)
    
        # Calculate sentiment for title and text (----- Not currently used in feature vectore -------)
        sent_dict = metrics.load_dict(app_dir + "AFINN-111.txt")
        two_grams = metrics.parse_dict(sent_dict)
        
        #num_chars, title_sent = metrics.calc_sentiment(title, sent_dict, two_grams)
        num_chars, txt_sent = metrics.calc_sentiment(url_txt, sent_dict, two_grams)   
        
        # Sent time created, time of day, and day of week
        #create_time = time.time()
        #create_tod = int(datetime.datetime.fromtimestamp(int(create_time)).strftime("%H"))
        #create_dow = dows[datetime.datetime.fromtimestamp(int(create_time)).strftime("%A")]    
        
        feature_list = ['Yups', 'title_sim']; 
        Xp_title = title_vect.transform([title])
        Xp_text = title_vect.transform([url_txt])
        
        test_list = [[int(time.time())], [num_chars], [txt_sent]]          
        
        Y_pred = model.test_ups(theta, test_list, train_set, Xp_title, Xp_text, X_title, X_text, feature_list, 5)
        
        score = int(round(10**Y_pred[0] - 1))
        percentage = 20        
        
        keywords = model.find_keywords(url_txt, X_title, train_set.loc[:,'Yups'].values, title_vect, 10)

        sim_inds, sim_vals = model.find_similar_posts(title_vect.transform([url_txt]), X_text, 3)
        sim_posts = train_set.loc[sim_inds, ['title', 'created_utc', 'permalink', 'Yups']]
    
        sim_posts['sim_vals'] = np.around(sim_vals,decimals=2).T

        percentage, peak, width = model.return_percentage(score, subreddit)       
        
    else:
        score = 0
        percentage = 0
        peak = 0
        keywords = []
        sim_posts = []
        
    return {"subreddit":subreddit, "score":score, "percentage":percentage, "peak": peak, "keywords":keywords, "posts":sim_posts}