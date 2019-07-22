# UNISDR Project - Multilabel Classification

## Description

This repository contains UNISDR machine learning project code used within the Emerging Technologies Team. This project attempts to improve the efficiency of classifying articles on UNISDR website for Office for Disaster Risk Reduction. This part of code is mainly to retrain the models when recieving the new datasets.

Currently the main focus is to allow:  

* The web prevention tool to use new articles to retrain the hazard models;
* The web prevention tool to use new articles to retrain the theme models; 

The following are the main features of the this codebase:

* Loading data and labels;
* Data Preprocessing and data transformation;
* Feature selection;
* Model selection, model evaluation and parameter adjustment;
* Save the trained models into disk;

## Dependencies

There are a number of project dependencies required to develop and operate the UNISDR Web Prevention Tool.

The following list details project dependencies:

* (IDE) Visual Studio Code Version 1.31;
* (PACKAGE) pyenchant 2.0.0;
* (PACKAGE) nltk 3.4;
* (CORPUS) nltk stopwords corpus;
* (PACKAGE) pandas 0.24.1;
* (PACKAGE) pathos 0.2.3;
* (PACKAGE) multiprocess 0.70.7;
* (PACKAGE) numpy 1.16.1;
* (PACKAGE) scikit-learn 0.20.1;
* (PACKAGE) textblob 0.15.2;
* (CORPUS) textblob corpus;
* (PACKAGE) re build-in;
* (PACKAGE) logging build-in;
* (PACKAGE) pickle build-in.

## Getting Started

* Clone the repository into your machine;
* Install all the above dependencies with correct Python version;
* Create "models" and "vector_models", "normalizar_models"  and "dim_reductor_models" folder in both ./unisdr/hazard and /unisdr/theme path and the models will automatically saved into these folder.
* Run training_container.py to get the text classification output;

## Documentation

Please refer to the following documents for this project:
* Project Initiation Document (PID) located at <Team_Directory>/projects/<Project_Name>/documents/;
* Design Document located at <Team_Directory>/projects/<Project_Name>/documents/;
* Analysis Document located at <Team_Directory>/projects/<Project_Name>/documents/.

## Author
* OICT/PSGD/ETT | ett@un.org
