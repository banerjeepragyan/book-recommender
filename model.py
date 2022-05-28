import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']


counts1 = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 10].index)]
counts = ratings['bookRating'].value_counts()
ratings = ratings[ratings['bookRating'].isin(counts[counts >= 10].index)]


combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)


combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])

book_ratingCount = (combine_book_rating.
     groupby(by = ['bookTitle'])['bookRating'].
     count().
     reset_index().
     rename(columns = {'bookRating': 'totalRatingCount'})
     [['bookTitle', 'totalRatingCount']]
    )


rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')


pd.set_option('display.float_format', lambda x: '%.3f' % x)


popularity_threshold = 10
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')


combined = rating_popular_book.merge(users, left_on = 'userID', right_on = 'userID', how = 'left')

combined = combined.drop('Age', axis=1)


combined = combined.drop_duplicates(['userID', 'bookTitle'])
combined.to_csv('combined_saved.csv')
combined_pivot = combined.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
combined_matrix = csr_matrix(combined_pivot.values)

print ("Model Training Started")

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(combined_matrix)

filename = 'finalized_model.sav'
pickle.dump(model_knn, open(filename, 'wb'))

print ("Model Training Done")