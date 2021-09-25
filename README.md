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
In order to run the codes successfully, you'll need to install below packages:
<br/> 
* pandas (1.1.*)
* numpy (1.18.5)
* nltk (3.5)
* sqlalchemy (1.3.18)
* pickle (4.0)
* sklearn (0.23.1)
<br/>

### File Descriptions
There are 3 folders in this repo. 
<br/> 
* data - ETL pipeline: This is the first step of the project, which consists of source data, database, and the script for data cleaning and preparation for the model training
* models - Machine Learning pipeline: This is the model training step of the project, including a script taking the cleaned data from ETL pipeline and training the models for tag prediction, and the best model based on grid search result saved as a pickle file
* app: This is the visualization step of the entire project.
* Please follow the steps in *Instructions* 
<br/>

There is one notebook for the codes used to perform the analysis and 3 csv files downloaded from Inside Airbnb as the data sources. Listings.csv and listings_sum.csv are both snapshots of listed resources as of the time the data was collected where listings.csv contains more information than the other. Reviews_sum.csv is a list of reviews with their objects and date only. There's also a reviews.csv dataset with more comprehensive information but we won't need it for this anlysis. Other pictures are output from the codes and are used in the published analysis for demonstration purposes. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
