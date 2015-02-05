from flask import render_template, request, redirect
import app_pca as app_v0
from app import app

#from .forms import EntryForm

@app.route('/')
@app.route('/index')
@app.route('/reddit')
def reddit():
    return render_template("reddit.html", samples = app_v0.samples(3))

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
@app.route('/reddit', methods=['GET', 'POST'])
def reddit_post():
    subdict = {0:"Other", 1:"/r/technology", 2:"/r/futurology", 3:"/r/gadgets"}
    url_post = request.form['url_text']
    if url_post:
	try:
            model_out = app_v0.fit_url(request.form['title_text'], url_post)
            return render_template('reddit.html', url_in = url_post, model_out = model_out, sub_txt = subdict[model_out["subreddit"]], samples = app_v0.samples(3), error = 0)
	except:
	    return render_template("reddit.html", samples = app_v0.samples(3), error = 1)
        return render_template('reddit.html', url_in = url_post, model_out = model_out, sub_txt = subdict[model_out["subreddit"]], samples = app_v0.samples(3), error = 0)
    else:
        return render_template("reddit.html", samples = app_v0.samples(3), error = 0)


@app.route('/slides', methods=['GET', 'POST'])
def slides():
    return render_template("slides.html")

@app.route('/plots', methods=['GET', 'POST'])
def plots():
    return render_template("plots.html")

if __name__ == '__main__':
    app.run()
#    app.run(host="0.0.0.0", port = 5000)
