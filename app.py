import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

with open('tfidf_matrix.pickle', 'rb') as B:
    tfidf_matrix = pickle.load(B)

with open('books.pickle','rb') as B:
    books = pickle.load(B)

st.title('Book Recommendation System')
st.write('Type a book name and get similar book recommendations')

def recommend(book_name, top_n=10):
    
    book_name = book_name.lower().strip()

    matches = books[books['title'].str.lower().str.contains(book_name, na=False)]
    if matches.empty:
        return 'Book not Found'

    idx = matches.index[0]

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    top_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
    recommendations = books.iloc[top_indices][['title','authors',]]
    return recommendations

book_list = books['title'].values
selected_book = st.selectbox('Select a book', book_list)

if st.button('Recommend'):
    recommendations = recommend(selected_book)
    if recommendations is None:
        st.error('Book Not Found!')
    else:
        st.write('Recommended Books:')
        st.dataframe(recommendations)