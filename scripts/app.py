import streamlit as st
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from config import output_directory, text_directory
from search_engine import load_documents as search_load_documents, calculate_tfidf, search, plot_similarity_scores
from topic_modeling import preprocess, load_documents as topic_load_documents, create_dictionary_corpus, apply_lda_model, extract_topics, calculate_coherence, prepare_visualization

st.set_page_config(layout="wide")

# Navigation
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ("Overview", "TF-IDF Matrix", "Search Engine", "Topic Modeling"))

################################## Overview Page ##################################
if choice == "Overview":
    st.title("Welcome to DataScience@Syracuse 🎓")
    
    st.write("""
    ## Overview 🌟
    
    This application is designed to assist prospective students interested in the Master of Science in Applied Data Science (MSADS) program at Syracuse University. Here's what you can do with the app:
    
    - **Explore the Curriculum:** Navigate through the program's curriculum to understand the courses offered.
    - **Search Functionality:** Use the app's search engine to find courses that match your interests, such as those involving Python programming.
    - **Discover Key Topics:** Utilize topic modeling to uncover the main topics covered across different courses.
    - **Analytical Insights:** Access a TF-IDF (Term Frequency-Inverse Document Frequency) matrix to dive deep into the textual analysis of the curriculum content.
    
    We hope this app makes it easier for you to explore what our MSADS program has to offer. Dive in and discover courses that align with your interests and career goals!
    """)

################################## TF-IDF Matrix Page ##################################

if choice == "TF-IDF Matrix":
    st.title("DataScience@Syracuse: TF-IDF Matrix DataFrame")
    
    # Read the uploaded CSV file into a DataFrame
    tfidf_matrix_df = pd.read_csv(f'{output_directory}/tfidf_matrix_df.csv')
    
    # Display the DataFrame using Streamlit's dataframe component
    st.dataframe(tfidf_matrix_df)


################################## Search Engine Page ##################################

elif choice == "Search Engine":
    st.title("DataScience@Syracuse: Search Engine")
    query = st.text_input("Enter your search query")

    if st.button("Search"):
        try:
            documents, filenames = search_load_documents(text_directory)
            vectorizer, tfidf_matrix = calculate_tfidf(documents)

            results = search(query, vectorizer, tfidf_matrix, filenames)
            results_df = pd.DataFrame(results, columns=['Document', 'Similarity'])
            st.data_editor(data=results_df, use_container_width=True, disabled=True)

            # Plotting
            fig = plot_similarity_scores(results, query)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error during search: {e}")

################################## Topic Modeling Page ##################################

elif choice == "Topic Modeling":
    st.title("DataScience@Syracuse: Topic Modeling")
    num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=4)

    if st.button("Perform Topic Modeling"):
        documents = topic_load_documents(text_directory)
        dictionary, corpus = create_dictionary_corpus(documents)
        lda_model = apply_lda_model(corpus, dictionary, num_topics=num_topics)
        
        topic_df = extract_topics(lda_model)
        coherence_lda = calculate_coherence(lda_model, documents, dictionary)
        vis = prepare_visualization(lda_model, corpus, dictionary)

        # Displaying results
        st.data_editor(data=topic_df,use_container_width=True, disabled=True)
        st.write('Coherence Score: ', coherence_lda)

        # Get the HTML visualization code
        vis = prepare_visualization(lda_model, corpus, dictionary)
        pyldavis_html = pyLDAvis.prepared_data_to_html(vis)

        # Modify the HTML to set width to 100%
        pyldavis_html = pyldavis_html.replace(
            '<div id="ldavis_el"', 
            '<div style="width: 100%;" id="ldavis_el"'
        )
        
        # Use the HTML in Streamlit with the full width of the container
        st.components.v1.html(pyldavis_html, height=800)

