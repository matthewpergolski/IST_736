# %%
import pandas as pd
import os
from glob import glob
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Import directory paths from secret config file
from config import pdf_directory, text_directory, output_directory

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    # Replace or remove unwanted characters
    text = text.replace('\x0c', '')  # Removes the form feed character
    # Add more replacements if needed
    return text

# Save extractions from PDF to a text file
def save_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

# Lemmatizer
def lemmatized_tokenizer(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha() and len(word) > 3]
    return lemmatized_tokens

# Main function
def main():
    # Make directories if they do not yet exist
    os.makedirs(text_directory, exist_ok=True)

    # Process each PDF file
    for pdf_path in glob(os.path.join(pdf_directory, '*.pdf')):
        pdf_text = extract_text_from_pdf(pdf_path)
        # Create a text file name based on the PDF file name
        text_file_name = os.path.splitext(os.path.basename(pdf_path))[0] + '.txt'
        text_file_path = os.path.join(text_directory, text_file_name)
        save_text_to_file(pdf_text, text_file_path)

    # Use glob to find all text files in the directory
    file_paths = glob(f'{text_directory}/*')
    file_paths

    # Initialize an empty dictionary to store the content of each file
    corpus_content = {}

    # Read each file and store its content in the dictionary
    for file_path in file_paths:
        file_name = file_path.split('/')[-1]
        with open(file_path, 'r') as file:
            corpus_content[file_name] = file.read()

    # Convert the dictionary into a DataFrame
    corpus_df = pd.DataFrame(list(corpus_content.items()), columns=['Label', 'Text'])
    print(corpus_df.info())
    corpus_df.head()

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=lemmatized_tokenizer)

    # Fit and transform
    X_vectorized = vectorizer.fit_transform(corpus_df['Text'])

    # Get dataframe
    tfidf_df = pd.DataFrame(X_vectorized.todense(), columns=vectorizer.get_feature_names_out(), index=corpus_df['Label'])
    tfidf_df.reset_index(inplace=True)

    # Pattern to extract class type (IST / SCM) followed by course number
    pattern = r'([A-Z]{3})\W*(\d{3})'

    # Replace current label text with regex extraction
    tfidf_df['Label'] = tfidf_df['Label'].str.extract(pattern).agg(' '.join, axis=1)

    print(tfidf_df.info(show_counts=True, verbose=True))
    tfidf_df

    tfidf_df.to_csv(f'{output_directory}/tfidf_matrix_df.csv', index=False)

# %%
if __name__ == "__main__":
    main()
