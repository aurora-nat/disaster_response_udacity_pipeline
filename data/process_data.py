import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """"
    This method takes in the messages csv file and categories csv
    and will load it into a dataframe.
    Parameters:
        messages_filepath: the location of the messages csv file to be loaded
        categories_filepath: the location of the categories csv file to be loaded
    Returns:
        A dataframe which merges the two datasets
    """
    # load the messages dataset
    messages = pd.read_csv(messages_filepath)

    # load the categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge the datasets
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    This method takes the previously created dataframe and cleans the data
    to be prepared for the machine learning pipeline. It will create the 36 classes
    and drop unneccessary columns. It will also remove any duplicated rows found
    in the dataset

    Parameters: dataframe of the merged categories and messages dataset

    Returns: a dataframe cleaned and ready for the machine learning model
    """
    # create col names from first row
    col_names = [item.split('-')[0] for item in df['categories'][0].split(';')]
    # create a dataframe of the 36 individual category columns
    df[col_names] = df.categories.str.split(';', expand=True)
    # remove original categories column
    df.drop(columns=['categories'], inplace=True)
    

    count = 0
    # remove the column name and just leave count
    for column in df:
        # skip over ID/Message/Original/Genre
        if count > 3:
            # first remove the column name
            df[column] = df[column].apply(lambda x: str(x[-1:]))
            # ensure column is an integer value
            df[column] = df[column].apply(lambda x: int(x))
        count += 1
    # drop duplicates
    df.drop_duplicates(inplace=True)
    #maintain data as binary - first locate any rows with extra class
    cat_col = df.columns[4:] #reference only categorical columns
    non_binary = ((df[cat_col].values)>= 2).any(1) #rows with more than 2 classes
    df = df[~non_binary] #remove these from our clean dataframe
    
    return df


def save_data(df, database_filename):
    """
        This metehod will save the cleaned dataset into an sqlite database.

        Parameter:
        df: the cleaned dataframe that will be saved as an sqlite database
        database_filename: the name we will give our new sqlite database
        Returns:
            None
    """
    engine = create_engine(database_filename, echo=True)
    # establish and sqllite connection:
    sql_conn = engine.connect()
    # pandas dataframe to sqllite database, create a categories table
    df.to_sql('categories', sql_conn, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
