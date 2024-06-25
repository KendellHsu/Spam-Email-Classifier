# AUTHOR: Kendell Hsu
import PySimpleGUI as sg
from load_classifier import load_spam_classifier
import joblib;

# Load the models and their accuracies
model_names = ['MultinomialNB', 'SVC', 'LogisticRegression', 'RandomForest']
models = {}
accuracies = {}

for model_name in model_names: #
    vectorizer, classifier = load_spam_classifier(model_name)
    models[model_name] = (vectorizer, classifier)
    accuracies[model_name] = joblib.load(f'{model_name}_accuracy.joblib')  # Load accuracies

# Create the GUI
def create_window():
    sg.theme('DarkBlue')  # You can change the theme as needed

    layout = [
        [sg.Text('Email Classifier', size=(30, 1), font=("Helvetica", 24), justification='center')],
        [sg.Text('Enter your email text:', size=(30, 1), font=("Helvetica", 20))],
        [sg.Multiline(size=(80, 15), key='-EMAIL-', font=("Helvetica", 20))],
        [sg.Text('Select algorithms:', size=(30, 1), font=("Helvetica", 20)),
         sg.Checkbox('MultinomialNB', default=True, key='-MULTINB-', font=("Helvetica", 20)),
         sg.Checkbox('SVC', default=True, key='-SVC-', font=("Helvetica", 20)),
         sg.Checkbox('LogisticRegression', default=True, key='-LOGREG-', font=("Helvetica", 20)),
         sg.Checkbox('RandomForest', default=True, key='-RF-', font=("Helvetica", 20))],
        [sg.Button('Classify', font=("Helvetica", 20), size=(10, 1)), 
         sg.Text('', key='-RESULT-', size=(50, 1), font=("Helvetica", 20))],
        [sg.Button('Detail', visible=False, font=("Helvetica", 20), size=(10, 1))],
        [sg.Text('', key='-DETAIL-', size=(80, 15), font=("Helvetica", 20), visible=False)]
    ]
    
    window = sg.Window('Spam Classifier', layout, size=(1200, 900), element_justification='center')
    
    results = []

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'Classify':
            email_text = values['-EMAIL-']
            selected_algorithms = []
            if values['-MULTINB-']:
                selected_algorithms.append('MultinomialNB')
            if values['-SVC-']:
                selected_algorithms.append('SVC')
            if values['-LOGREG-']:
                selected_algorithms.append('LogisticRegression')
            if values['-RF-']:
                selected_algorithms.append('RandomForest')
            
            results = []
            
            for model_name in selected_algorithms:
                vectorizer, classifier = models[model_name]
                email_vector = vectorizer.transform([email_text])
                prediction = classifier.predict(email_vector)[0]
                results.append((model_name, prediction))
            
            # Weighted vote
            weights = {name: accuracies[name] for name in selected_algorithms}
            spam_score = sum(weights[name] for name, pred in results if pred == 1)
            ham_score = sum(weights[name] for name, pred in results if pred == 0)
            final_result = 'Spam' if spam_score > ham_score else 'Ham'
            
            # Update result text with color
            if final_result == 'Spam':
                window['-RESULT-'].update(final_result, text_color='red')
            else:
                window['-RESULT-'].update(final_result, text_color='green')
            
            window['Detail'].update(visible=True)
            window['-DETAIL-'].update(visible=False)
        
        elif event == 'Detail':
            if results:
                detail_text = "\n".join([f"{name}: {'Spam' if pred == 1 else 'Ham'}" for name, pred in results])
                window['-DETAIL-'].update(detail_text, visible=True)
    
    window.close()

if __name__ == "__main__":
    create_window()