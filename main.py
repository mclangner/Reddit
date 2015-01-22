import json
import random
import time
import datetime
import urllib2,cookielib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymysql as mdb
from bs4 import BeautifulSoup

dows = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}

def data_to_df(post, subred, rank, read_time):
    # take some relevant info from post, convert to data frame
    outp = {}
    
    grab = ['id', 'created_utc', 'domain', 'url', 'title', 'permalink', 
    'score', 'ups', 'link_flair_text', 'num_comments']       
    
    for key in grab:
        grab_val = post[key]
        if grab_val is None:
            outp[key] = 'None'
        else:
            if type(grab_val) == unicode:
                outp[key] = post[key].encode('latin-1', 'ignore')
            else:
                outp[key] = post[key]
    
    create_time = post['created_utc']
    
    outp['subred'] = subred
    outp['rank'] = rank
    outp['active'] = 1
    outp['read_time'] = read_time
    outp['create_tod'] = int(datetime.datetime.fromtimestamp(int(create_time)).strftime("%H"))
    outp['create_dow'] = dows[datetime.datetime.fromtimestamp(int(create_time)).strftime("%A")]
    outp['sentiment'] = 0
    outp['age'] = read_time - create_time
    outp['txt_dl'] = 0
    outp['com_dl'] = 0

    return pd.DataFrame(outp, index = [1])

def read_posts(url, subred, read_time):
    req = urllib2.Request(url) 
    req.add_header("User-agent", "Matts tech reader by u/matt_the_physicist")
    page=json.load(urllib2.urlopen(req))
    
    data = page[u'data'][u'children']
    
    if subred > 0:
        rtop = 9
    else:
        rtop = 1
    
    count = 0
    for i in range(rtop):
        print i, ' ' + url
        aft_txt = page[u'data'][u'after']
        count = count + 100
        time.sleep(random.random() * 10 + 10)
        load_url = url + "&count=" + str(count) + "&after=" + aft_txt
        print load_url
        req = urllib2.Request(load_url) 
        req.add_header("User-agent", "Matts tech reader by u/matt_the_physicist")
        page=json.load(urllib2.urlopen(req))
        data = data + page[u'data'][u'children']

    print 'here 1'
    df = pd.DataFrame()
    count = 0
    for item in data:
        count = count + 1
        cur_post = data_to_df(item['data'], subred, count, read_time)        
        
        df = df.append(cur_post, ignore_index = True)
        
    print 'here 2'
        
    return df

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
    print url
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
        except urllib2.HTTPError as err:
            #txt = err.fp.read()
            txt = "HTTPError"
            status = 1
        except urllib2.URLError as err:
            txt = "URLError"
            status = 1
                
    return txt, status
    
def download_pages(dtype):
    con = mdb.connect('localhost', 'pyconnect', 'pypa$$1', 'Reddit')    

    if dtype in [0,2]:
        strSQL = "Select id, url from PostData Where txt_dl = " + str(dtype) + "0;"
        outdf = pd.io.sql.read_sql(sql = strSQL, con = con)
        
        print outdf.shape
    
        for post in outdf.iterrows():
            print post
            pid = post[1]['id']
            purl = post[1]['url']
            
            print len(purl)
        
            post_txt, status = get_text_from_url(purl)
        
            if status == 0:
                txtfile = open('text/' + pid, 'w')
                txtfile.write(post_txt)
                txtfile.close
                 
                with con:
                    cur = con.cursor()
                    cur.execute("UPDATE PostData SET txt_dl = 1 WHERE id = '" + pid + "';")
            else:
                print purl
                print post_txt
                
                with con:
                    cur = con.cursor()
                    cur.execute("UPDATE PostData SET txt_dl = 2 WHERE id = '" + pid + "';")

def backup_csv(fname1, fname2):
    con = mdb.connect('localhost', 'pyconnect', 'pypa$$1', 'Reddit')    

    strSQL = "Select * from PostData;"
    
    outdf = pd.io.sql.read_sql(sql = strSQL, con = con)
    outdf.to_csv(fname1)
    
    strSQL = "Select * from RankData;"
    
    outdf = pd.io.sql.read_sql(sql = strSQL, con = con)
    outdf.to_csv(fname2)

def main_grabber():
        
    #subreddit numbers: tech = 1, futurology = 2, gadgets = 3
    
    tech_urls = ["http://www.reddit.com/r/technology/.json?limit=100", 
    "http://www.reddit.com/r/futurology/.json?limit=100", 
    "http://www.reddit.com/r/gadgets/.json?limit=100"]
    
    
    other_urls = ["http://www.reddit.com/r/music/.json?limit=100", 
    "http://www.reddit.com/r/worldnews/.json?limit=100", 
    "http://www.reddit.com/r/sports/.json?limit=100",
    "http://www.reddit.com/r/movies/.json?limit=100"]
        
    post_data_labels = ['id', 'active', 'created_utc', 'domain', 'link_flair_text', 
    'permalink', 'sentiment', 'subred', 'title', 'url', 'txt_dl', 'com_dl', 'create_dow', 'create_tod']

    post_dynamic_labels = ['id', 'num_comments', 'rank', 'age', 'ups']
    
    con = mdb.connect('localhost', 'pyconnect', 'pypa$$1', 'Reddit')     

    for (ind, url) in enumerate(tech_urls):
        cur_df = read_posts(url, ind+1, int(time.time()))
        print 'here 3'
        post_data = cur_df.loc[:, post_data_labels]
        print 'here 4'
        rank_data = cur_df.loc[:, post_dynamic_labels]
        print 'here 5'
        with con:
            cur = con.cursor()
            cur.execute("DELETE FROM NewPosts;")    
        post_data.to_sql(con=con, name='NewPosts', if_exists='append', flavor = 'mysql')

        #check for new posts        
        str_SQL = "SELECT A.* from NewPosts A LEFT JOIN PostData B on (A.id = B.id) Where B.id Is Null;"        
        
        new_posts = pd.io.sql.read_sql(sql = str_SQL, con=con)
        print "%d new posts in url" %new_posts.shape[0]
        new_posts = new_posts.loc[:,post_data_labels]
        new_posts.to_sql(con=con, name='PostData', if_exists='append', flavor='mysql')
        rank_data.to_sql(con=con, name='RankData', if_exists='append', flavor='mysql')
       
        time.sleep(random.random() * 15 + 15)
        
    for (ind, url) in enumerate(other_urls):
        cur_df = read_posts(url, 0, int(time.time()))
        print 'here 3'
        post_data = cur_df.loc[:, post_data_labels]
        print 'here 4'
        rank_data = cur_df.loc[:, post_dynamic_labels]
        print 'here 5'
        with con:
            cur = con.cursor()
            cur.execute("DELETE FROM NewPosts;")        
        
        post_data.to_sql(con=con, name='NewPosts', if_exists='append', flavor = 'mysql')
        
        #check for new posts
        str_SQL = "SELECT A.* from NewPosts A LEFT JOIN PostData B on (A.id = B.id) Where B.id Is Null;"        
        
        print str_SQL
        new_posts = pd.io.sql.read_sql(sql = str_SQL, con=con)
        print "%d new posts in url" %new_posts.shape[0]
        new_posts = new_posts.loc[:,post_data_labels]
        new_posts.to_sql(con=con, name='PostData', if_exists='append', flavor='mysql')
        rank_data.to_sql(con=con, name='RankData', if_exists='append', flavor='mysql')
        
        time.sleep(random.random() * 15 + 15)
#        

# next return urls for un-downloaded text, and download
# then check if 
    
            