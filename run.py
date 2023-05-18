import re
import json
import plotly
import nltk
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from flask import Flask
from flask import render_template, request
import joblib
from sqlalchemy import create_engine
from plotly.graph_objects import Bar



app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True


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
    # Tokenize the text into a list of words
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    # We lemmatize our tokens to get them reduced to the base of the word
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('categories', engine)


# load model
# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    categories = {}
    # We will get the counts for each of the categories to understand where
    # the data is coming from (genre)
    for category in df.columns[4:]:
        labeled_count = {}
        count = df[category].value_counts()
        i = 0
        # add the genre label to the counts for each category
        for v in count:
            if i == 0:
                labeled_count.update({'direct': v})
            if i == 1:
                labeled_count.update({'news': v})
            if i == 2:
                labeled_count.update({'social': v})
            i += 1

            categories.update({category: labeled_count})
    # create visuals for all our graphs
    graphs = []

    for key in categories:
        # create new x,y values for each key in categories
        x_val = []
        y_val = []
        # iterate through the second dict to get the counts
        genre = categories[key]
        for name in genre:
            # x is composed of the genre names
            x_val.append(name)
            # y is composed of the genre count for the specific category
            y_val.append(genre[name])
        # after this loop is complete, we can create a bar graph for this
        # specific category
        graph = [
            {
                'data': [
                    Bar(
                        x=x_val,
                        y=y_val
                    )
                ],

                'layout': {
                    'title': 'Distribution of Message for {}'.format(key),
                    'yaxis': {
                        'title': "Count"
                    },
                    'xaxis': {
                        'title': "Genre"
                    }
                }
            }
        ]
        # add our created graph and continue to add for all categories
        graphs.extend(graph)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    #app.run(host='0.0.0.0', port=3001, debug=True)hero
    app.run(debug=True)


if __name__ == '__main__':
    main()
