# Chicago Car Crashes

## Overview
* Purpose: Create a Classifier to determine the primary cause of a car crash in the city of Chicago
* Source of data: Sourced data from Chicago Government website
* Data excluded: Any crashes where the primary cause of the accident was unknown or unreported.

## Data Analysis & Recommendations
* Close to 50% of crashes in Chicago have an unknown primary cause of accident

<img src = '/images/business_problem.png'>

* The best model, after tuning, was our random forest model

<img src = '/images/cm_randomforest.png'>

<img src = '/images/roc_randomforest.png'>

* The most important features based on our model:

<img src = '/images/feature_importance.png'>

## Recommendations

* The model provides a helpful tool for investigating car crashes where the primary cause is unknown.
* Prioritize investigations based on important features highlighted by the model.

# Repository Navigation
Our Github Repository contains 2 main folders names Data and Notebooks. The Notebooks folder has 3 jupyter notebooks in it. The first of those notebooks is titled Data_Cleaning and contains all of our data cleaning and exploration steps. It is important to run this notebook first as cleaned data csv files are created which the Modeling notebook then calls. The next notebook is titled Modeling and contains all of the models we created as well as evaluation for each model as well. The Data folder contains all of the raw data we looked at as well as the cleaned data we ended using reconverted back into csv files. 

# Presentation and Sources
Presentation: [Link](https://docs.google.com/presentation/d/1-mDPeNw8ceEtgrEbTC_9TN8NCnfrG85J4SGQtEGcP_g/edit#slide=id.g2b2d0d004c9_0_564)

Crashes dataset: [Link](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if/about_data)

People dataset: [Link](https://data.cityofchicago.org/Transportation/Traffic-Crashes-People/u6pd-qa9d)

Vehicles dataset: [Link](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Vehicles/68nd-jvt3/about_data)
