#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    UNISDR Prevention Web Text Classification Solution

    This module is the entry point for the multi label text classification
    problem for UNISDR
"""

import time                                         # Calculcate time differences
import re                                           # Package used to as replace function
import numpy as np                                  # Mathematic caculations
import pandas as pd                                 # Dataframe operations
from helper import Helper as ett_h                  # ETT Helper methods
from helper import InvalidLabelsToModelsError       # Custom ETT Exception
from transformer import Transformer as ett_t        # ETT Transformer methods
from constants import RegexFilter                   # Regex Filter Enums
from constants import Language                      # KV Language Enums
from enum import Enum                               # Custom enum for classification type
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import os

# Custom UNISDR Classification Type (only two currently)
class ClassificationType(Enum):
    HAZARD = "HAZARD",
    THEME = "THEME"


# Text Classification Class
class ModelTraining:


    # Global variable declaration
    data = pd.DataFrame()  # dataframe
    label = []  # list

    # Processing variables
    num_of_labels = 0  # number
    
    # File and path based variables
    base_folder_location = os.path.dirname(os.path.abspath(__file__))
    label_file_name = 'labels.txt'
    model_folder_name = 'models'
    vector_folder_name = 'vector_models'
    dim_reductor_folder_name = 'dim_reductor_models'
    normalizar_folder_name = 'normalizar_models'
    model_name = '_model.pickle'
    vector_model_name = '_vectorizer.pickle'
    dim_reductor_model_name = '_dim_reductor.pickle'
    normalizar_model_name = '_normalizar.pickle'

    ##
    # UNISDR application constructor
    # @param models Tuple of TextModel objects
    # @param data DataFrame of the data to be processed
    # @param labels Tuple of strings of the expected labels
    # @raise InvalidLabelsToModelsError custom ETT exception for incorrect label / model sizes
    def __init__(self, data, labels, label_type):
        self.data = data
        self.labels = labels
        self.num_of_labels = len(labels)
        self.load_label()
    
    # Entry point method to actually start the
    # classification operations
    def process(self):
        self.training_model()
    
    def load_labels(self):  # The output is a list
        abs_filename = ett_h.generate_dynamic_path([self.base_folder_location, self.label_type.lower(), self.label_file_name])
        self.label = (ett_h.load_data_common_separated(abs_filename, ','))
        self.num_of_labels = len(self.label)

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.label, test_size=0.2)

    # Method which acts as the builder #Automatically select 
    def training_model(self):
        # Split the data into 0.8 training datasets and 0.2 testing datasets
        self.train_test_split(self.data, self.label, 0.2)
        pipeline = imbPipeline([
                        ('tfidf', TfidfVectorizer()),
                        ('oversample', SMOTE(random_state=42)),
                        ('svd', TruncatedSVD()),
                        ('nor',preprocessing.MinMaxScaler()),
                        ('clf', OneVsRestClassifier(SVC()))])

        #list_c = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
        list_c = [0.5,1]
        #list_c =  np.linspace(0,1,5)
        list_n =[100,110]
        # Remember to add[2,\]2]
        best_score = 0
        epsilon = .005
        dictionary = {}
        for para_c in list_c:
            for para_n in list_n:
                parameters = {'tfidf':[TfidfVectorizer(max_features = 800,ngram_range=(1,4), norm='l2', encoding='latin-1',stop_words='english', analyzer='word')],
                            'svd':[TruncatedSVD(n_components=para_n, n_iter=7, random_state=42)],
                            'clf':[OneVsRestClassifier(SVC(kernel='linear',probability=True,C = para_c))]
                            }
                gs_clf = GridSearchCV(pipeline, parameters,cv=5,error_score='raise',n_jobs = -1)
                gs_clf = gs_clf.fit(X_train, y_train)
                current_score = gs_clf.best_score_
                dictionary[current_score] = parameters

        for current_score in dictionary.keys():
            if current_score - epsilon > best_score:
                best_score = current_score
                print(best_score)
        model_dict = dictionary[best_score]
        
        abs_filename_m = ett_h.generate_dynamic_path([self.base_folder_location, self.label_type.lower(), self.model_folder_name, label + self.model_name]) 
        abs_filename_v = ett_h.generate_dynamic_path([self.base_folder_location, self.label_type.lower(), self.vector_folder_name, label + self.vector_model_name]) 
        abs_filename_r = ett_h.generate_dynamic_path([self.base_folder_location, self.label_type.lower(), self.dim_reductor_folder_name, label + self.dim_reductor_model_name]) 
        abs_filename_n = ett_h.generate_dynamic_path([self.base_folder_location, self.label_type.lower(), self.normalizar_folder_name, label + self.normalizar_model_name]) 
        
        # Here to fit the training datasets to the  models with best score
        # vectorization
        vector = model_dict['tfidf'][0].fit(self.X_train, self.y_train)
        ett_h.save_model(abs_filename_v)
        
        # Balcancing
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(vectorized_df, y_train)
        
        # Feature selction
        svd = model_dict['svd'][0].fit(X_res, y_res)
        ett_h.save_model(abs_filename_r)

        # Normalizing
        min_max_scaler = preprocessing.MinMaxScaler()
        nor_model = min_max_scaler.fit(dim_reductor_df, y_res)
        ett_h.save_model(abs_filename_n)

        # Classifier
        clf = model_dict['clf'][0].fit(scaled_df, y_res)
        clf.fit(scaled_df, y_res)
        ett_h.save_model(abs_filename_m)