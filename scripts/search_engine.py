#%%
import pandas as pd
import os
import re
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# Import directory paths from secret config file
from config import text_directory

# Preprocessing the text
def preprocess(text):
    # Convert to lowercase, remove punctuation, and split into words
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Function to extract class type (IST / SCM) followed by course number
def extract_course_label(filename):
    match = re.search(r'([A-Z]{3})\W*(\d{3})', filename)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return filename

# Load documents and create a TF-IDF representation
def load_documents(directory):
    documents = []
    filenames = []
    for filename in os.listdir(directory):
        # Apply the regex pattern to each filename
        course_label = extract_course_label(filename)
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
            documents.append(preprocess(text))
            filenames.append(course_label)
    return documents, filenames

# Calculate TF-IDF and return the vectorizer and document matrix
def calculate_tfidf(documents):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

# Implementing the search function with cosine similarity
def search(query, vectorizer, tfidf_matrix, filenames):
    query_vec = vectorizer.transform([preprocess(query)])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Ranking documents by similarity
    ranked_indices = np.argsort(cosine_similarities)[::-1]
    ranked_filenames = [(filenames[i], cosine_similarities[i]) for i in ranked_indices if cosine_similarities[i] > 0]

    return ranked_filenames

# Generate plotly visualization
def plot_similarity_scores(search_results, query):
    df = pd.DataFrame(search_results, columns=['Document', 'Similarity'])
    fig = px.bar(df, x='Document', y='Similarity', 
                 title=f'Document Similarity Scores for "{query}"', 
                 color='Document',
                 color_discrete_sequence=px.colors.qualitative.G10)

    fig.update_layout(showlegend=False)
    return fig

# Main function
def main():
    # Get documents and filenames
    documents, filenames = load_documents(text_directory)
    df = pd.DataFrame({'File Names': filenames,
                        'Documents' : documents})

    print(df.info())
    df.head()

    # Calculate TF-IDF matrix from the documents
    vectorizer, tfidf_matrix = calculate_tfidf(documents)

    # Define a query term
    query = "Jupyter"
    print(f'Query: {query}\n')

    # Search for the query in the TF-IDF matrix and get similarity scores
    search_results = search(query, vectorizer, tfidf_matrix, filenames)
    print(search_results)

    # Create a DataFrame to display documents and their similarity scores
    df_similarity = pd.DataFrame(search_results, columns=['Document', 'Similarity'])

    # Output the basic information of the DataFrame
    print(df_similarity.info())

    # Show the first few rows of the DataFrame
    df_similarity.head()

    # Visualization with Plotly: Plot the similarity scores for each document
    fig = plot_similarity_scores(df_similarity, query)
    fig

#%%
if __name__ == "__main__":
    main()