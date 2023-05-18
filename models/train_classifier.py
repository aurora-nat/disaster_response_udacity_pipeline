import sys
import re
import pandas as pd
import pickle
import nltk
import numpy as np
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Original author: Udacity
    Transformer utilized to extract the verb from the sentence. This helps
    increase the meaning behind the context of the message being utilized. 
    """
    def starting_verb(self, text):
        #tokenize per sentence
        sentence_list = nltk.sent_tokenize(text)

        for sentence in sentence_list:
            #add a part of speech tag to our tokenized sentence
            pos_tags = nltk.pos_tag(tokenize(sentence))
            
            if len(pos_tags) > 1:
                #grab first word/tag
                first_word, first_tag = pos_tags[0]
                #if first tag is appropriate verb
                if first_tag in ['VB', 'VBP']:
                    return 1
                else:
                    return 0
            else:
                return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        '''
            The transform method will apply the starting verb to all values in X
            and transform the series into a dataframe to be utilized later, the dataframe
            ensures no nan values are found

            Parameters: 
                X: The series message we are looking at that we will transform
            Returns: 
                X_taged: The new dataframe that has been tagged
        '''
        X_tagged = pd.Series(X).apply(self.starting_verb)
        X_tagged = pd.DataFrame(X_tagged)
        #RandomForestClassifier cannot accept missing values, so we must ensure they are droppped
        X_tagged = X_tagged.replace(np.nan, 0)
        return X_tagged

def load_data(database_filepath):
    """
    This method will load a dataframe from the database engine SQLite
    through the database filename

    Parameters:
        database_filepath: the path of the database

    Returns:
        The dataframe feature and label arrays and category names

    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('categories', engine)
    # get a list of all the category names:
    category_names = list(df.columns[4:])
    # Split our data for X, Y values
    X = df['message']
    y = df.loc[:, 'related':'direct_report']
    return X, y, category_names


def tokenize(text):
    """
        This method will tokenize the text that we will analyze later with
        our NLP Machine Learning Model. This method normalizes the text first,
        tokenizes it using the NLTK library and then lemmatizes the text

        Parameters: The text for our disaster response message

        Returns: A list of clean tokens for our Machine Learning Model


    """
    # Normalize the text first removing punctuation and making text lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize the text into a list of wordsß
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
   
    # We lemmatize our tokens to get them reduced to the base of the word
    clean_tokens = [lemmatizer.lemmatize(token.strip()) for token in tokens]
  
    return clean_tokens


def build_model():
    """
        Builds our machine learning pipeline and sets
        a parameter grid, where the GridSearchCV hypertunes
        the set of parameters to determine the best possible model

        Parameters: None

        Returns: A gridsearch representation of our Machine Learning Pipeline
        for the final model pipeline


    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                    ('tfidf_vect', TfidfVectorizer(tokenizer=tokenize)),
                    ('std', StandardScaler(with_mean=False))
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),
        
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__tfidf_vect__ngram_range': [(1, 1)],
        'features__text_pipeline__tfidf_vect__norm': [None],
        'features__text_pipeline__tfidf_vect__max_features': ([1000]),
        'clf__estimator__n_estimators': [300,400],
        'clf__estimator__min_samples_split': [2]


    }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This method performs our machine learning model's
    evaluation of the performance based on a classification report.

    Parameters:

        model: a gridsearchCV model created through our build_model method earlier
        X_test: the train-test split for the testing feature variable dataset (messages)
        Y_test: the train-test split of the testing target variable dataset (36 classes)
        category_names: the target's 36 class labels

    Returns: None
        However, outputs the model's classification report for each class.
        And prints the model's overal accuracy score.
    """

    # create a list of target names for the classification report
    target_names = category_names
    # Find the predicted values of y
    y_pred = model.predict(X_test)

    # To iterate the columns of the predicated values we have to transform our
    # numpy array into dataframes:
    y_pred_df = pd.DataFrame(y_pred, columns=target_names)
    y_test_df = pd.DataFrame(Y_test, columns=target_names)

    # print report for each col:
    for col in y_pred_df.columns:
        y_true = y_test_df[col]

        y_pred_i = y_pred_df[col]

        report = classification_report(y_true, y_pred_i)
        print(f"Classification report for {col}:")
        print("/n================================")
        print(report)
        print('====================================/n')
    print("")
    # Print the model's overall Score

    score = model.score(X_test, Y_test)
    print("Accuracy score of our model: {:.2f}%".format(score * 100))


def save_model(model, model_filepath):
    """
        This method will save our model to a pickle file so that it can
        be utilized by our web application.

        Parameters:
            model: the finalized model that we will be using in our web app
            model_filepath: the location where the pickle file will be stored
        Returns: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
