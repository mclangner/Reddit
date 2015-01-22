import re
import string
import pandas as pd
import pymysql as mdb
import numpy as np
import matplotlib.pyplot as plt

def load_dict(filename):
    
    sent_file = open(filename, 'r')
    scores = {}
    for line in sent_file:
        term, score = line.split("\t")
        scores[term] = int(score)
        
    sent_file.close
    return scores

def parse_dict(dict_in):
# returns n > 1 grams in sentiment dictionary
    two_grams = {}    
    for term in dict_in:
        if len(term.split()) > 1:
            two_grams[term] = dict_in[term]
    
    return two_grams

def calc_sentiment(tstr, sent_dict, two_grams):
    sent_score = 0
    num_chars = 0    
    
    #find n > 1 grams in string, score, and delete them
    for gram in two_grams:
        if gram in tstr:
            gram_count = tstr.count(gram)
            sent_score = sent_score + two_grams[gram]*gram_count
            num_chars = num_chars + gram_count*len(gram.replace(' ', ''))
            tstr.replace(gram, '')
            
    tlist = tstr.split()
            
    # This line splits on all punctuation - not necessary if already stripped
    #tlist = re.findall(r"[\w']+", txt)

    # now step through text looking for 1-grams    
    for item in tlist:
        num_chars = num_chars + len(item)
        if item in sent_dict:
            sent_score = sent_score + sent_dict[item]
            
    return num_chars, sent_score

def calc_metrics():
    con = mdb.connect('localhost', 'pyconnect', 'pypa$$1', 'Reddit')    

    #Select articles that have downloaded text and length hasn't been calculated
    strSQL = "Select id, title from PostData where num_chars is Null and txt_dl = 1";

    outdf = pd.io.sql.read_sql(sql = strSQL, con = con)

    sent_dict = load_dict('AFINN-111.txt')
    two_grams = parse_dict(sent_dict)

    # make exclude set for stripping punctuation            
    exclude = set(string.punctuation)    
    
    for post in outdf.iterrows():
        txtfile = open("text/" + post[1]['id'], 'r')
        txt = txtfile.read()
        txtfile.close()

        # strip punctuation
        txt = ''.join(ch for ch in txt if ch not in exclude)
        
        num_chars, txt_sent = calc_sentiment(txt, sent_dict, two_grams)
        title_len, title_sent = calc_sentiment(post[1]['title'], sent_dict, two_grams)
        print post[1]['id'], num_chars, txt_sent, title_sent
        
        with con:
            strSQL = "UPDATE PostData SET num_chars = " + str(num_chars) + " "
            strSQL = strSQL + ", txt_sent = " + str(txt_sent) + " "
            strSQL = strSQL + ", title_sent = " + str(title_sent) + " "
            strSQL = strSQL + "WHERE id = '" + post[1]['id']+ "';"
            cur = con.cursor()
            cur.execute(strSQL)
            
def plot_histogram(subred, field, fn = "MAX"):
    con = mdb.connect('localhost', 'pyconnect', 'pypa$$1', 'Reddit')      
    
    #Select metric from RankData for given subreddit by joining with PostData
    strSQL = "Select A.id, B.subred, " + fn + "("+ field + ") as metric FROM RankData A INNER JOIN PostData B ON (B.id = A.id) "
    strSQL = strSQL + "GROUP BY A.id HAVING B.subred = " + str(subred) + ";"
    outdf = pd.io.sql.read_sql(sql = strSQL, con = con)  
   
    data = outdf.loc[:,'metric'].values

    data = np.log10(data+1) 
    
    hist, bins = np.histogram(data, bins=25)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    
    histdf = pd.DataFrame([bins, hist])
    histdf = histdf.T
    histdf.columns = ['bins', 'count']
    
    #plt.bar(center, hist, align='center', width=width)
    plt.plot(center, np.log10(hist+1), '*', markersize = 12)
    plt.show()

    return histdf
    
    
        