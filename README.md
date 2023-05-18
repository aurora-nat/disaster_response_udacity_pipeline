##  Udacity Disaster Response Pipeline

The purpose of this project was to analyze disaster data from Appen and build a machine learning model API that classifies disaster messages. The model creates a Natural Language Processor Pipeline which is used to classify the disaster within a set of 36 classes (a mutlioutputClassifier). The project is designed using bootstrap and allows an emergency worker to type in the incoming emergency message to understand the type of emergency we are dealing with. This type of application can positively impact various emergency teams to reduce time spent categorizing disasters. This would increase efficiency in being able to send appropriate resources to the disaster locations.


## Files

- app: 
    - templates:
        - go.html: classification result page of web app
        - master.html: main page of web app
    - run.py: Flask file that runs app
- data:
    - disaster_categories.csv: data that will be processsed  
    - disaster_messages.csv: data to process
    - process_data.py: our data is processed, cleaned, and saved.
    - DisasterResponse.db - database that has been saved & clean
- models:
    - train_classifier.py: contains code for our NLP pipeline
    - classifier.pky: saved model
- README.md
- requirements.txt

## Project Files

- app: 
    - templates:
        - go.html: classification result page of web app
        - master.html: main page of web app
    - run.py: Flask file that runs app
- data:
    - disaster_categories.csv: data that will be processsed  
    - disaster_messages.csv: data to process
    - process_data.py: our data is processed, cleaned, and saved.
    - DisasterResponse.db - database that has been saved & clean
- models:
    - train_classifier.py: contains code for our NLP pipeline
    - classifier.pky: saved model
- README.md
- requirements.txt

### Instructions to run code locally:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        ``` sh
        python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv sqlite:///data/DisasterResponse.db
        ```
    - To run ML pipeline that trains classifier and saves
        ``` sh 
        python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl 

        ```

2. Run the following command in the app's directory to run your web app.
    ``` sh 
    python3 run.py
     ```

3. Go to http://0.0.0.0:3001/


## Note

  The models: classifier.pkl was unsupported due to GitHub sizing limit, however, runnning `process_data.py + models/train_classifier.py`
    Will generate your own saved model on your local computer. 






