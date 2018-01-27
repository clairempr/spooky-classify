# Use scikit-learn and spaCy to classify text for
# Kaggle Spooky Author Identification competition
# Claire Pritchard, January 2018

import csv
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as Pipeline_imb
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
import spacy

nlp = spacy.blank('en')  # create blank Language class
spacy_tokenizer = nlp.Defaults.create_tokenizer(nlp)


def main():
    data_dir = 'data'
    training_file = 'train.csv'
    test_file = 'test.csv'
    model_file = 'text_clf.pkl'

    # explore data, fit, evaluate, save model
    fit_and_save_model(model_file, data_dir, training_file)

    # load saved model and make predictions
    load_model_and_predict(model_file, data_dir, test_file)


# explore data, fit, evaluate, save model
def fit_and_save_model(model_file, data_dir, training_file):
    # read data from csv to get text and labels
    df = read_csv(data_dir, training_file)
    X = df['text']
    y = df['author']

    # show distribution of number of samples
    show_data_plot(y)

    # get pipeline for vectorizing, transforming, and fitting data
    text_clf = make_pipeline()

    # split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Cross-validation
    cross_validate(text_clf, X, y, cv=5)

    # fit and score the model
    train_and_evaluate(text_clf, X_train, X_test, y_train, y_test)

    # show confusion matrix and classification report
    show_confusion_matrix(text_clf, X_test, y_test)

    # refit model with all the data
    print('Fitting model with all the data')
    text_clf = text_clf.fit(X, y)

    # save model
    save_model(text_clf, model_file)


# load saved model and make predictions
def load_model_and_predict(model_file, data_dir, test_file):
    # load previously trained model
    text_clf = load_model(model_file)

    # make predictions for kaggle test data
    make_predictions(data_dir, test_file, text_clf)


# read csv into dataframe
def read_csv(data_dir, filename):
    df = pd.read_csv(os.path.join(data_dir, filename))

    return df


def make_pipeline():
    # Combine over- and under-sampling using SMOTE and Tomek
    res = SMOTETomek()
    mnb = MultinomialNB(alpha=0.01, fit_prior=False)
    bnb = BernoulliNB(alpha=.0001)
    mlp = MLPClassifier(hidden_layer_sizes=20, alpha=0.001, activation='relu', verbose=True)
    vcl = VotingClassifier(estimators=[
        ('mnb', mnb),
        ('bnb', bnb),
        ('mlp', mlp)],
        voting='soft')

    text_clf = Pipeline_imb([
        ('feats', FeatureUnion([
                        # Pipeline for word counts and tfidf
                        ('vect_tfidf', Pipeline([
                            ('vect', CountVectorizer(analyzer='word',
                                                     ngram_range=(1, 2),
                                                     lowercase=False,
                                                     stop_words=None,
                                                     tokenizer=Tokenizer())),
                            ('tfidf', TfidfTransformer(use_idf=False)),
                        ])),
                        # Pipeline for calculating extra features from text
                        ('body_stats', Pipeline([
                            ('stats', TextStats()),
                            ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                        ])),

            ])),
        ('res', res),
        ('scaler', MaxAbsScaler(copy=True)), # helps mnb a little, helps mlp a tiny bit, no effect on bnb?
        ('clf', vcl),
        ])

    return text_clf


def save_model(text_clf, filename):
    print('Saving trained model')
    joblib.dump(text_clf, filename)


def load_model(filename):
    print('Loading saved model')
    return joblib.load(filename)


# fit and score the model
def train_and_evaluate(text_clf, X_train, X_test, y_train, y_test):
    print('Training model')
    text_clf = text_clf.fit(X_train, y_train)
    print('Score on training data: {}'.format(text_clf.score(X_train, y_train)))
    # evaluate
    print('Score on test data: {}'.format(text_clf.score(X_test, y_test)))


# Cross-validation
def cross_validate(text_clf, X, y, cv):
    print('Cross-validating...')
    cv_scores = cross_val_score(text_clf, X, y, cv=cv,
                                # n_jobs=-1,
                                verbose=3)

    print('Cross-validation scores:', cv_scores)
    print('Mean cross-validation score: {:.3f}'
          .format(np.mean(cv_scores)))


def make_predictions(data_dir, test_file, text_clf):
    print('Writing predictions file')
    df = read_csv(data_dir, test_file)
    with open('predictions.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"')
        csvwriter.writerow(['id', 'EAP', 'HPL', 'MWS'])
        probabilities = text_clf.predict_proba(df['text'])
        for id, prob in zip(df['id'], probabilities):
            csvwriter.writerow([id] + list(prob))


def show_data_plot(y):
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(unique, counts)
    plt.show()


def show_confusion_matrix(text_clf, X_test, y_test):
    predicted = text_clf.predict(X_test)
    confusion = confusion_matrix(y_test, predicted)
    df_cm = pd.DataFrame(confusion,
                         index=[i for i in range(0, 3)], columns=[i for i in range(0, 3)])

    plt.figure(figsize=(5.5, 4))
    sns.heatmap(df_cm, annot=True)
    plt.title('Classifier \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, predicted)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # classification report
    print('Classification report:')
    print(classification_report(y_test, predicted,))

    plt.show()


class TextStats(BaseEstimator, TransformerMixin):
    """Takes in sentence, extracts number of tokens"""

    def __init__(self):
        pass

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        return [{
                'sentence_length': len(tokens),
                'std_word_length': np.std([len(token) for token in tokens]),

            }
            for tokens in [tokenize(text, lemmatize=False) for text in X]]

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class Tokenizer(object):
 def __call__(self, text):
     return tokenize(text, lemmatize=True)


def tokenize(text, lemmatize):
    if lemmatize:
        # lemmatized tokens
        tokens = [token.lemma_ for token in spacy_tokenizer(text)]
    else:
        # tokens
        tokens = [token.text for token in spacy_tokenizer(text)]
    return tokens


main()
