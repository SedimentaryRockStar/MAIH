from flask import *
from joblib import load
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn import model_selection, naive_bayes, svm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

clf = load("classifier.joblib")


Corpus = pd.read_csv("preprocessed.csv", encoding='latin-1')
Encoder = LabelEncoder()
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['Sentence'], Corpus['Emotion'], test_size=0.3)

Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(Corpus['Sentence'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

app = Flask(__name__)
#SESSION_TYPE = 'filesystem'
#app.config.from_object(__name__)
app.secret_key = "abc"  


# Base endpoint to perform prediction.
@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    X = Tfidf_vect.transform([text])
    a = clf.predict(X)
    prediction = Encoder.inverse_transform(a)    
    return render_template("prediction.html", prediction=prediction[0])
