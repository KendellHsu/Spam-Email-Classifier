# load_classifier.py
import joblib
import os

def load_spam_classifier(model_name):
    vectorizer_path = f'{model_name}_vectorizer.joblib'
    classifier_path = f'{model_name}_classifier.joblib'
    
    if os.path.exists(vectorizer_path) and os.path.exists(classifier_path):
        vectorizer = joblib.load(vectorizer_path)
        classifier = joblib.load(classifier_path)
    else:
        raise FileNotFoundError(f"Model files not found. Please run train_classifier.py to train and save the model {model_name}.")
    
    return vectorizer, classifier
