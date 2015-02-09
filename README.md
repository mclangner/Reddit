<article class="markdown-body entry-content" itemprop="mainContentOfPage">

<strong>Project Description for the Blogtimize/r/</strong>
</h1>

<h3>
<a id="user-content-problem-and-goals" class="anchor" href="#problem-and-goals" aria-hidden="true"><span class="octicon octicon-link"></span></a>Problem and Goals</h3>

<p> The <a href ="http://www.blogtimize.me">Blogtimize/r/</a>, created as a demo project for Insight Data Science, aims to find an appropriate audience for tech-related blog posts and scores those posts by relevance to the target audience. This is done by comparing the content to posts that have been posted on reddit.</p>

<h3>
Data Collection</h3>

<p>
Article titles, link URLs, post times, vote numbers, and site ranking were collected using the reddit API. The program then follows the URLs to the articles and scrapes the main text of the using Beautiful Soup. Separate word vectors are assembled using the sklearn tf-idf.

Articles were collected from /r/technology, /r/futurology, /r/gadgets, /r/worldnews, /r/music, /r/movies, and /r/sports.
</p>

<h3>
Algorithm - PCA Branch</h3>
<p>
Posts are classified in two steps using a naive Bayes algorithm, and the number of votes are estimated using a random forest regressor. The first classification step sorts posts as belonging to the tech-related subreddits (technology, futurology, gadgets), or not. If the post is tech-related, it is then sorted into one of the three tested subreddits. Classifying in two steps increases the accuracy of classification.

The program uses tf-idf word vectors for the naive Bayes classification steps, and the performs a principle component analysis to reduce the dimensionality of the word feature vectors in the random forest regressor step. Time of day, day of the week, sentiment score for the article text, sentiment score for the article title, and article length are additional features for the regression step. 

In each step (two classifier steps and random forest regression), the word vectors are reconstructed, creating a more specialized vocabulary for subsequent steps.</p>

<h3>
Algorithm - Master Branch</h3>

<p>The master branch algorithm is the same for the classification step but differs in the regression step. The master code uses the number of votes, age, and length of the 10 most similar posts as the features for the radnom forest regressor. Currently, the PCA feature vector gives more accurate results on cross-validation tests.</p>

</article>
