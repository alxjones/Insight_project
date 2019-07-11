#! /usr/bin/env python
from flask import Flask, request, render_template, redirect, url_for
from IsMyBabyWeird import app
from sklearn.externals import joblib
import praw
from praw.models import MoreComments
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))

# Set up reddit API authentication
reddit = praw.Reddit(client_id='My ID',
                     client_secret='My Secret',
                     user_agent='User agent')
# Get subreddits to scrape

subredditsToScan = reddit.subreddit('Parenting+daddit+beyondthebump')
# clf = joblib.load('/home/ubuntu/application/IsMyBabyWeird/static/clf.joblib')
with open('/home/ubuntu/application/IsMyBabyWeird/static/clf', 'rb') as fp:
    clf = pickle.load(fp)
with open('/home/ubuntu/application/IsMyBabyWeird/static/vocab', 'rb') as fp:
    vocab = pickle.load(fp)



def query_web(query):
    topics_dict = {"id": [], "title": [], "body": []}
    comment_dict = {"post": [], "author": [], "body": [], "score": [], "parent_post": []}

    search = subredditsToScan.search(query=query, limit=100)
    for submission in search:
        topics_dict["id"].append(submission.id)
        topics_dict["title"].append(submission.title)
        topics_dict["body"].append(submission.selftext)
    if not topics_dict['id']:
        return "No results found"

    else:
        topics_data = pd.DataFrame(topics_dict)
        topics_data['token_body'] = topics_data['body'].apply(word_tokenize)
        topics_data['token_body'] = topics_data['token_body'].apply(lambda x: [w for w in x if w not in stop_words])
        topics_data['token_body'] = topics_data['token_body'].apply(" ".join)
        topics_data = topics_data[topics_data['token_body'].str.contains(str(query), case=False)]

        for post in topics_data["id"]:
            submission = reddit.submission(id=post)
#            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                if isinstance(comment, MoreComments):
                    continue
                comment_dict["post"].append(comment.link_id)
                comment_dict["author"].append(comment.author)
                comment_dict["body"].append(comment.body)
                comment_dict["score"].append(comment.score)
                comment_dict["parent_post"].append(submission.selftext)
        comment_df = pd.DataFrame(comment_dict)
        comment_df.to_csv('/home/ubuntu/application/IsMyBabyWeird/static/temp.csv')
        return comment_df


def standardize_text(df, text_field):
    df['std_text'] = df[text_field].str.replace(r"http\S+", "")
    df['std_text'] = df[text_field].str.replace(r"http", "")
    df['std_text'] = df[text_field].str.replace(r"@\S+", "")
    df['std_text'] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df['std_text'] = df[text_field].str.replace(r"@", "at")
    df['std_text'] = df[text_field].str.lower()
    return df



def cv(data):
    count_vectorizer = CountVectorizer(vocabulary=vocab)
    emb = count_vectorizer.fit_transform(data)
    return emb, count_vectorizer


def predict_relevance(df):
    standardized = standardize_text(df, 'body')
    list_corpus = standardized['std_text'].tolist()
    counts, count_vectorizer = cv(list_corpus)
    labels = clf.predict(counts)
    decis_fxn = clf.decision_function(counts)
    df['predicted'] = labels
    df['rank'] = decis_fxn
    df['post_rank'] = df.groupby('post')['rank'].transform(np.mean)
    df = df[df['predicted'] == 1]
    df = df.sort_values('post_rank', ascending=False)
    df.reset_index()
    parents = df['post'].unique()
    return df, parents


@app.route('/', methods=["GET", "POST"])
def index():
    to_screen = None
    if request.method == "POST":
        query = request.form.get("u_query")
        query = word_tokenize(query)
        query = [w for w in query if not w in stop_words]
        query = " ".join(query)
        comments = query_web(query=query)
        if type(comments) == str:
            to_screen = comments
        else:
            return redirect(url_for('results'))

    return render_template("base.html", relevant=to_screen)


@app.route('/results', methods=["GET", "POST"])
def results():
    comments = pd.read_csv('/home/ubuntu/application/IsMyBabyWeird/static/temp.csv')
    relevant, parents = predict_relevance(comments)
    option_list = parents.tolist()
    to_screen = None
    if request.method == "POST":
        parent_post = request.form.get("post_option")
        to_screen = relevant[relevant['post'] == parent_post].parent_post.iloc[0]
        for i in range(0, len(relevant[relevant['post'] == parent_post])):
            to_screen += '<p class="tab">' + relevant[relevant['post'] == parent_post].body.iloc[i] + '</p>'
            to_screen += '<br>'
            to_screen += ''
            i += 1
    else:
        to_screen = relevant[relevant['post'] == option_list[0]].parent_post.iloc[0]
        for i in range(0, len(relevant[relevant['post'] == option_list[0]])):
            to_screen += '<p class="tab">' + relevant[relevant['post'] == option_list[0]].body.iloc[i] + '</p>'
            to_screen += '<br>'
            to_screen += ''
            i += 1

    return render_template("result.html", option_list=option_list, relevant=to_screen)

