import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import torch
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import joblib
import pandas as pd

st.write("# Book Recommender")

col1, col2, col3 = st.columns(3)

combined = pd.read_csv('combined_saved.csv')
combined_pivot = combined.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
combined_matrix = csr_matrix(combined_pivot.values)

bookname = []
for i in range(combined_pivot.shape[0]):
    bookname.append(combined_pivot.index[i])

rcm = []

def foundbook (indexn):
    query_index = indexn
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    distances, indices = loaded_model.kneighbors(combined_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)

    combined_pivot.iloc[query_index,:].values.reshape(1,-1)

    for i in range(0, len(distances.flatten())):
        if i == 0:
            print()
            rcm.append('Recommendations :\n'.format(combined_pivot.index[query_index]))
        else:
            rcm.append('{0}: {1}'.format(i, combined_pivot.index[indices.flatten()[i]]))

#name = input("Enter book name: ").strip()

name = col1.text_input("Enter a book you have read recently:").strip()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

if st.button('Predict'):

    bk = ""

    corpus_embeddings = embedder.encode(bookname, convert_to_tensor=True)

    queries = [name]

    top_k = min(1, len(bookname))
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        for score, idx in zip(top_results[0], top_results[1]):
            bk = bookname[idx]

    indexn = bookname.index(bk)

    foundbook(indexn)

    for s in rcm:
        st.write(s)

    for s in rcm:
        print(s)

print ("END")
