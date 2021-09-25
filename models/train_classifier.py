import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split, GridSearchCV


def load_data(database_filepath):
    '''
    Read in cleaned data from database and output model ready training data X, Y and list of categories we are prediction
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('data_cleaned', engine)  
    X = df['message'].values
    Y = df.iloc[:,4:].values
    Y = np.delete(Y, 9, 1) # remove the col with only 0
    
    labels = list(df.columns[4:9]) + list(df.columns[10:])
    
    return X, Y, labels


def tokenize(text):
    '''
    Tokenize and lemmatize the input text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize and remove punctuation and stopwords
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() 
                        for tok in tokens 
                        if tok.isalpha() and tok not in stopwords.words('english')]
    return clean_tokens

def build_model():
    '''
    Create pipeline and parameters for grid search

    OUTPUT: defined model with hyperparameter to be tunned
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression(multi_class='ovr')))
    ])

    parameters = {
        'clf__estimator__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'], 
        'clf__estimator__C': [10, 1.0, 0.1]
    }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, verbose=12, n_jobs=-1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluating the trained model using test dataset and print out results
    '''
    
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(category_names[i], ':')
        print(classification_report(Y_test[:, i], Y_pred[:, i]))
        

def save_model(model, model_filepath):
    '''
    Save the trained model as a pickle file
    '''
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()