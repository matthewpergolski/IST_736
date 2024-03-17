#%%
import pandas as pd
import os
import re
import nltk
import gensim
import pyLDAvis.gensim_models as gensimvis
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel

# Import directory paths from secret config file
from config import text_directory

# Download NLTK resources (this can be done outside the functions, as a setup step)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Removes stopwords, lemmatizes, and cleans document text
def preprocess(document):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = re.sub(r'\W+', ' ', document.lower()).split()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha() and len(token) > 3]
    return tokens

# Loads and preprocesses all documents from a given directory
def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            documents.append(preprocess(file.read()))
    return documents

# Creates a dictionary and corpus from preprocessed documents for LDA analysis
def create_dictionary_corpus(documents):
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    return dictionary, corpus

# Builds and returns an LDA model from the corpus and dictionary
def apply_lda_model(corpus, dictionary, num_topics=4):
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=2)
    return lda_model

# Extracts and formats topics and their respective words and weights from the LDA model
def extract_topics(lda_model):
    def parse_topic_words(topic_str):
        word_weight_pairs = topic_str.split(' + ')
        parsed_pairs = [pair.split('*') for pair in word_weight_pairs]
        return [(float(weight.strip()), word.strip('"')) for weight, word in parsed_pairs]

    topic_data = []
    for idx, topic in lda_model.print_topics(-1):
        for weight, word in parse_topic_words(topic):
            topic_data.append([idx, word, weight])

    topic_df = pd.DataFrame(topic_data, columns=['Topic', 'Word', 'Weight'])
    return topic_df.sort_values(by=['Topic', 'Weight'], ascending=[True, False])

# Calculates and returns the coherence score of the LDA model
def calculate_coherence(lda_model, documents, dictionary):
    coherence_model_lda = CoherenceModel(model=lda_model, texts=documents, dictionary=dictionary, coherence='c_v')
    return coherence_model_lda.get_coherence()

# Prepares data for visualization of the LDA model
def prepare_visualization(lda_model, corpus, dictionary):
    return gensimvis.prepare(lda_model, corpus, dictionary)

# Main function
def main():
    # Load and preprocess documents
    documents = load_documents(text_directory)
    print(f"Loaded {len(documents)} documents.\n")

    # Create dictionary and corpus
    dictionary, corpus = create_dictionary_corpus(documents)
    print("Dictionary and corpus created.\n")

    # Apply LDA model
    num_topics = 4
    lda_model = apply_lda_model(corpus, dictionary, num_topics)
    print(f"LDA model with {num_topics} topics applied.\n")

    # Extract topics
    topic_df = extract_topics(lda_model)
    print(topic_df.info())
    topic_df

    # Create grouped dataframe along with dictionairy object for filtering
    groups = topic_df.groupby('Topic')
    topic_dfs = {}

    for topic, group in groups:
        topic_dfs[topic] = group

    for t in topic_dfs.keys():
        print(f'{topic_dfs[t]}\n')

    # Calculate coherence
    coherence_score = calculate_coherence(lda_model, documents, dictionary)
    print(f"Coherence score: {coherence_score}\n")

    # Run prepare_visualization function
    prepare_visualization(lda_model, corpus, dictionary)

#%%
if __name__ == "__main__":
    main()
