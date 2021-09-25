# Disaster Response Pipeline Project

### Installation
In order to run the codes successfully, you'll need to install below packages:
<br/> 
* pandas (1.1.*)
* numpy (1.18.5)
* nltk (3.5)
* sqlalchemy (1.3.18)
* pickle (4.0)
* sklearn (0.23.1)
)<br/>

After that the code should run with no issues using Python versions 3.*.

### Project Summary
This project is a disaster response pipeline that aims to predict relevant disaster categories from social media messages, in order to help first responders to quickly identify the issue right after a disaster happened.

### File Descriptions
There are 3 folders in this repo. 
<br/> 
* data - ETL pipeline: This is the first step of the project, which consists of source data, database, and the script for data cleaning and preparation for the model training
* models - Machine Learning pipeline: This is the model training step of the project, including a script taking the cleaned data from ETL pipeline and training the models for tag prediction, and the best model based on grid search result saved as a pickle file
* app: This is the visualization step of the entire project.
* Please follow the steps in *Instructions* to run the entire pipeline on your local computer.
<br/>

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
