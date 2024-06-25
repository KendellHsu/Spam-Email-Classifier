import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_dataset import load_and_preprocess_data
from each_method import train_naive_bayes, train_svc, train_logistic_regression, train_random_forest

def train_and_save_model(model_func, model_name):
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_extraction = load_and_preprocess_data()
    
    # Train the model
    model = model_func(X_train, y_train)
    
    # Evaluate the model on training data
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_precision = precision_score(y_train, train_predictions)
    train_recall = recall_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions)
    
    # Evaluate the model on testing data
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)
    
    # Print the evaluation metrics
    print(f"{model_name} - Train Accuracy: {train_accuracy}")
    print(f"{model_name} - Train Precision: {train_precision}")
    print(f"{model_name} - Train Recall: {train_recall}")
    print(f"{model_name} - Train F1 Score: {train_f1}")
    print(f"{model_name} - Test Accuracy: {test_accuracy}")
    print(f"{model_name} - Test Precision: {test_precision}")
    print(f"{model_name} - Test Recall: {test_recall}")
    print(f"{model_name} - Test F1 Score: {test_f1}")
    
    # Save the model, feature extractor, and accuracy
    joblib.dump(feature_extraction, f'{model_name}_vectorizer.joblib')
    joblib.dump(model, f'{model_name}_classifier.joblib')
    joblib.dump(test_accuracy, f'{model_name}_accuracy.joblib')

if __name__ == "__main__":
    classifiers = {
        'MultinomialNB': train_naive_bayes,
        'SVC': train_svc,
        'LogisticRegression': train_logistic_regression,
        'RandomForest': train_random_forest
    }
    
    for name, train_func in classifiers.items():
        train_and_save_model(train_func, name)                                                                                                                                                                                                                                  
        print(f"Model {name} trained, evaluated, and saved successfully.")