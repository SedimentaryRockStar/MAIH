import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from joblib import dump

np.random.seed(500)
Corpus = pd.read_csv(r"/Users/goudanhan/Downloads/mergedData4_delete_empty.csv", encoding='latin-1')

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(Corpus['Sentence'], Corpus['Emotion'], test_size=0.3)

Encoder = LabelEncoder()
Y_train = Encoder.fit_transform(Y_train)
Y_test = Encoder.fit_transform(Y_test)

'''
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]'''


Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(Corpus['Sentence'])
X_train_Tfidf = Tfidf_vect.transform(X_train)
X_test_Tfidf = Tfidf_vect.transform(X_test)

'''# fit the training dataset on the NB classifier
NaiveBayes = naive_bayes.MultinomialNB()
NaiveBayes.fit(X_train_Tfidf, Y_train)
# predict the labels on X_test_Tfidf
predictions_NB = NaiveBayes.predict(X_test_Tfidf)

# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Y_test))'''


# fit the training dataset on the SVM classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train_Tfidf, Y_train)
# predict the labels on X_test_Tfidf
predictions_SVM = SVM.predict(X_test_Tfidf)

'''# Test with user input
X = Tfidf_vect.transform([input("Input here: ")])
a = SVM.predict(X)
print(Encoder.inverse_transform(a))'''

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Y_test))

dump(SVM, 'classifier.joblib')

"""# fit the training dataset on the logistic classifier
logistic = LogisticRegression(random_state=0, max_iter=800)
logistic.fit(X_train_Tfidf, Y_train)
# predict the labels on X_test_Tfidf
predictions_logistic = logistic.predict(X_test_Tfidf)

# Use accuracy_score function to get the accuracy
print("Logistic Regression Accuracy Score -> ", accuracy_score(predictions_logistic, Y_test))"""
