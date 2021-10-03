import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

np.random.seed(500)
Corpus = pd.read_csv(r"/Users/goudanhan/Downloads/mergedData3.csv", encoding='latin-1')

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['Sentence'], Corpus['Emotion'], test_size=0.3)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(Corpus['Sentence'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
print("finished")
# fit the training dataset on the NB classifier
'''Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

X = Tfidf_vect.transform([input("Input here: ")])
a = SVM.predict(X)
print(Encoder.inverse_transform(a))

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)'''

# Classifier - Algorithm - Logistic Regression
# fit the training dataset on the classifier
logistic = LogisticRegression(random_state=0, max_iter=800)
logistic.fit(Train_X_Tfidf, Train_Y)
# predict the labels on validation dataset
predictions_logistic = logistic.predict(Test_X_Tfidf)
X = Tfidf_vect.transform([input("Input here: ")])
a = logistic.predict(X)
print(Encoder.inverse_transform(a))
# Use accuracy_score function to get the accuracy
print("Logistic Regression Accuracy Score -> ", accuracy_score(predictions_logistic, Test_Y) * 100)
