import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
dataset_path = 'dataset/Spam_Ham_Dataset.csv'  # Updated dataset path
df = pd.read_csv(dataset_path)

# Preprocess data
X = df['Message']  # Column containing the email text
y = df['Category']  # Column containing 'Spam' or 'Ham'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical format using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes model
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model and vectorizer for future use
with open('model/spam_classifier.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

with open('model/vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

# Define Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get email text from the form
        email_text = request.form['email']
        
        # Load the trained model and vectorizer
        with open('model/spam_classifier.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        
        with open('model/vectorizer.pkl', 'rb') as vec_file:
            loaded_vectorizer = pickle.load(vec_file)
        
        # Transform the input email text
        email_vec = loaded_vectorizer.transform([email_text])
        
        # Predict using the model
        prediction = loaded_model.predict(email_vec)
        
        # Return the result to the user
        return render_template('index.html', prediction=prediction[0], email=email_text)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
