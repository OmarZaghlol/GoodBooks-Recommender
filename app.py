## Backend packages
from flask import Flask, url_for, request, render_template, jsonify, json

## ML packages
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from cosine_sim import *

def recommend(title, n=10):
    title = title.lower()
    if title in titles.str.lower().unique():
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:31]
        book_indices = [i[0] for i in sim_scores]
        df = books.iloc[book_indices]

        v = df['ratings_count']
        m = df['ratings_count'].quantile(0.60)
        R = df['average_rating']
        C = df['average_rating'].mean()
        df['weighted_rating'] = (R*v + C*m) / (v + m)
        
        qualified = df[df['ratings_count'] >= m]
        qualified = qualified.sort_values('weighted_rating', ascending=False).head(n)

        # top10 = {'title':list(qualified['title'].values)
        #         ,'year':list(qualified['original_publication_year'].values)
        #         ,'rating':list(qualified['average_rating'].values)
        #         ,'image_url':list(qualified['image_url'].values)}

        top10 = {'id':[],'title':[],'author':[],'rating':[],'image_url':[]}
        url = 'https://www.goodreads.com/book/show/'
        for i in qualified.index:
            top10['id'].append(url+str(qualified.loc[i]['book_id']))
            top10['title'].append(qualified.loc[i]['title'])
            top10['author'].append(qualified.loc[i]['authors'])
            # top10['year'].append(int(qualified.loc[i]['original_publication_year']))
            top10['rating'].append(qualified.loc[i]['average_rating'])
            top10['image_url'].append(qualified.loc[i]['image_url'])

       
        return top10
    else:
        # in case of the book not in the database
        return "Sorry, that book isn't in our database"


app = Flask(__name__)

## Load the data 
books = pd.read_csv('data/books.csv')

## Load the Model
cosine_sim = cosine_sim(books)
# try:
# 	cosine_sim = joblib.load('models/cosine_sim.pkl')
# except:
# 	cosine_sim = cosine_sim(books)
# 	joblib.dump(cosine_sim, 'models/cosine_sim.pkl')


## Preprocess the data
# year: float-> int
books['original_publication_year'] = books['original_publication_year'].fillna(-1).apply(lambda x: int(x) if x != -1 else -1)
# put the titles in lower format for ease of search
# and save in a variable for not losing the right titles
indices = pd.Series(books.index, index=books['title'].str.lower())
titles = books['title']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    recommendations = []
    if request.method == 'POST':
        title = request.form['title']
        recommendations = recommend(title)
        
    # return jsonify(recommendations)
    return render_template('results.html', recommendations=recommendations, 
    						n=len(recommendations['id']))


if __name__ == '__main__':
    app.run(debug=True)
