import os

# Get the directory where the config.py file is located
scripts_dir = os.path.abspath(os.path.dirname(__file__))

# Go up one level to the root directory from the scripts directory
root_dir = os.path.join(scripts_dir, os.pardir)

# Directory containing PDFs, relative to the root directory
pdf_directory = os.path.join(root_dir, 'data', 'pdfs')

# Directory to save converted text files, relative to the root directory
text_directory = os.path.join(root_dir, 'data', 'pdfs_converted_to_text')

# Directory containing TFIDF Matrix Data Frame, relative to the root directory
output_directory = os.path.join(root_dir, 'data', 'output')
