
"""
This script downloads all the necessary NLTK data packages required for the 
YouTube Transcript Summarizer. Run this script once before starting the server.
"""

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    """Download required NLTK data packages."""
    packages = [
        'punkt',        # Tokenizer for breaking text into sentences
        'stopwords',    # Common words to filter out
        'wordnet',      # Lexical database for semantic similarity
        'averaged_perceptron_tagger'  # Part-of-speech tagger
    ]
    
    for package in packages:
        print(f"Downloading NLTK package: {package}")
        try:
            nltk.download(package)
            print(f"Successfully downloaded {package}")
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")
    
    print("\nNLTK setup complete!")

if __name__ == "__main__":
    print("Setting up NLTK data packages for YouTube Transcript Summarizer...")
    download_nltk_data()
    print("\nYou can now run the server with 'python server.py'")