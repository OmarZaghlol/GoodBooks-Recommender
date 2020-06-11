from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import pandas as pd

ratings = pd.read_csv('data/ratings.csv')
book_tags = pd.read_csv('data/book_tags.csv')
tags = pd.read_csv('data/tags.csv')

def get_genres(x):
	t = book_tags[book_tags.goodreads_book_id==x]
	return [i.lower().replace(" ", "") for i in tags.tag_name.loc[t.tag_id].values]


def cosine_sim(books):
	ratings_rmv_duplicates = ratings.drop_duplicates()
	unwanted_users = ratings_rmv_duplicates.groupby('user_id')['user_id'].count()
	unwanted_users = unwanted_users[unwanted_users < 3]
	unwanted_ratings = ratings_rmv_duplicates[ratings_rmv_duplicates.user_id.isin(unwanted_users.index)]
	new_ratings = ratings_rmv_duplicates.drop(unwanted_ratings.index)
	new_ratings['title'] = books.set_index('id').title.loc[new_ratings.book_id].values

	books['authors'] = books['authors'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x.split(', ')])
	books['genres'] = books.book_id.apply(get_genres)
	books['soup'] = books.apply(lambda x: ' '.join([x['title']] + x['authors'] + x['genres']), axis=1)
	count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
	count_matrix = count.fit_transform(books['soup'])
	cosine_sim = cosine_similarity(count_matrix, count_matrix)
	return cosine_sim