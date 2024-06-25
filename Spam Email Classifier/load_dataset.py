import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download NLTK resources
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# Paths to email datasets
legitimate_emails = 'data/legiti_emails/'
spam_emails = 'data/spam_emails/'

# Function to load the data
def get_data(path):
    data = []
    files = os.listdir(path)
    
    for file in files:
        with open(path + file, encoding="ISO-8859-1") as processed_file:
            words_list = processed_file.read()
            data.append(words_list)
    
    return data

# Function to clean emails
def clean_emails(emails):
    cleaned_emails = []
    
    for email in emails:
        lines = email.split('\n')
        content = ''
        for line in lines:
            if line.startswith('Subject:'):
                subject = line.replace('Subject:', '').strip()
            elif line.startswith('From:'):
                sender = line.replace('From:', '').strip()
            elif line.startswith('To:'):
                recipient = line.replace('To:', '').strip()
            elif line.startswith('Date:'):
                date = line.replace('Date:', '').strip()
            elif line.startswith('X-'):
                continue
            else:
                content += line.strip()

        cleaned_emails.append({'content': content})

    return cleaned_emails

# Function to preprocess text
def preprocess_text(text):
    def is_english_word(word):
        synsets = wordnet.synsets(word)
        return len(synsets) > 0 and synsets[0].lemmas()[0].name() == word.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text.lower())
    english_stopwords = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in english_stopwords]
    english_words = [token for token in filtered_tokens if is_english_word(token)]
    
    return ' '.join(english_words)

def load_and_preprocess_data():
    # Load emails
    ham = get_data(legitimate_emails)
    spam = get_data(spam_emails)

    # Clean the legitimate and spam emails
    cleaned_ham = clean_emails(ham)
    cleaned_spam = clean_emails(spam)

    # Convert to DataFrame
    legitimate = pd.DataFrame(cleaned_ham)
    spam = pd.DataFrame(cleaned_spam)

    # Apply preprocessing
    legitimate['content'] = legitimate['content'].apply(preprocess_text)
    spam['content'] = spam['content'].apply(preprocess_text)

    # Combine data for TF-IDF Vectorization
    legitimate['label'] = 'ham'
    spam['label'] = 'spam'
    data = pd.concat([legitimate, spam])

    # Extract features using TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english', min_df=1, lowercase=True)
    X = tfidf.fit_transform(data['content'])
    y = data['label'].map({'ham': 0, 'spam': 1}).astype(int)
    

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    
    return X_train, X_test, y_train, y_test, tfidf

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, tfidf = load_and_preprocess_data()
    print(f'Training data shape: {X_train.shape}')
    print(f'Testing data shape: {X_test.shape}')
    print('Data loaded and preprocessed successfully.')