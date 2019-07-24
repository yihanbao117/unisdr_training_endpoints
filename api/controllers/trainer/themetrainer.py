#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Acts as a testing mechanism for the UNISDR solution.

    The UNISDR solution utilizes the ETT core modules which
    are a set of modules of common/generalized functions to use
    with the ETT Data Science Standards.
"""

__author__ = "Yihan Bao"
__copyright__ =  "Copyright 2019, United Nations OICT PSGD ETT"
__credits__ = ["Kevin Bradley", "Yihan Bao", "Praneeth Nooli"]
__date__ = "22 July 2019"
__version__ = "0.1"
__maintainer__ = "Yihan Bao"
__email__ = "yihan.bao@un.org"
__status__ = "Development"

import os  # Package used for accessing files
import time  # Calculcate time differences
import pandas as pd  # Dataframe operations
import sys  # Pakcage for system operations
import json  # Package for josn file operation
sys.path.append('../ett/')  # Natigate to ett folder path
from helper import Helper as ett_h  # ETT Helper methods
from run import app  # From run.py import application
from transformer import Transformer as ett_t  # ETT Transformer methods
from constants import JobType  # Enum for job type
from constants import RegexFilter  # Regex Filter Enums
from constants import Language  # KV Language Enums 
from constants import LabelType  # ETT constants for label types
from cleanser import Cleanser as ett_c # ETT cleanser package
from sklearn.model_selection import train_test_split # Pakcage for data spitting 
from sklearn.decomposition import TruncatedSVD  # Pakcage for feature engineering
from sklearn import preprocessing  # Pakcage for preprocessing 
from sklearn.svm import SVC  # Pakcage for support vector machien model
from sklearn.feature_extraction.text import TfidfVectorizer  # Pakcage for tfidf vector
from sklearn.multiclass import OneVsRestClassifier  # Pakcage of onevsrest model
from sklearn.model_selection import GridSearchCV  # Pakcage for model selectiom
from imblearn.over_sampling import SMOTE  # Package for data balancing
from imblearn.pipeline import Pipeline as imbPipeline  # Package for pipline creation
from flask import request  # Flask methods for requesting binary input file
from flask_restful import Resource  # Flask restful for create endpoints

base_folder_location = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
label_file_name = 'labels.txt'
data_file_name = 'data.json'

model_folder_name = 'models'
vector_folder_name = 'vector_models'
dim_reductor_folder_name = 'dim_reductor_models'
normalizar_folder_name = 'normalizar_models'

model_name = '_model.pickle'
vector_model_name = '_vectorizer.pickle'
dim_reductor_model_name = '_dim_reductor.pickle'
normalizar_model_name = '_normalizar.pickle'

# Preprocessing variables
colnames = ["title", "textData"]

@app.route('/training/theme/data', methods=['POST'])
def training_theme_data():
    
    bytes_data = request.stream.read()
    bytes_data = ett_t.bytes_to_str(bytes_data)
    bytes_data = json.loads(bytes_data) 
    global input_data
    input_data = pd.DataFrame(bytes_data)
    return "Successfully uploading THEME data"

class TrainTheme(Resource):
    

    def post(self):

        # Get the THEME labels
        abs_filename = ett_h.generate_dynamic_path([base_folder_location, LabelType.THEME.value, label_file_name])
        labels = (ett_h.load_data_common_separated(abs_filename, ','))
        # Get the label data from input_data
        raw_label = input_data['label']
        data = ett_t.transform_data_to_dataframe_basic(input_data, colnames)
        # Get the OneHotEncoded labels
        label_df = ett_t.one_hot_encoding(raw_label)  #17 labels dataframe
        # Rename the OneHotEncoded labels
        label_df.columns = labels  
        # Get the number of labels
        num_of_labels = len(labels)      
        # Data preprocessing
        nan_cleaned_data = ett_c.clean_dataframe_by_regex(data, RegexFilter.NON_ALPHA_NUMERIC.value)  # Removed all non alphanumeric characters
        d_cleaned_data = ett_c.clean_dataframe_by_regex(nan_cleaned_data, RegexFilter.DIGITS_ONLY.value)  # Removed all digits
        l_cleaned_data = ett_c.remove_non_iso_words(d_cleaned_data, Language.ENGLISH.value)   # Remove non-English text
        rew_cleaned_data = ett_c.remove_language_stopwords(l_cleaned_data, Language.ENGLISH.name)  # Remove English stop words
        l_transformed_data = ett_t.lowercase(rew_cleaned_data)  # Transform text to lowercase
        le_transformed_data = ett_t.stemming_mp(l_transformed_data)  # Transform text to core words i.e. playing > play
        data = le_transformed_data  # Return the newly transformed data

        # Split the data into 0.8 training datasets and 0.2 testing datasets
        X_train, X_test, y_train, y_test = train_test_split(data, label_df, test_size=0.2, random_state=42) 
        best_score_list = [] 
        for i in range(num_of_labels):
            single_label = y_train.iloc[:,i]
            label = labels[i]
            print("label",label)
            pipeline = imbPipeline([
                            ('tfidf', TfidfVectorizer()),  # Data vectorization
                            ('oversample', SMOTE(random_state=42)),  # Data balancing
                            ('svd', TruncatedSVD()),  # Feature selection
                            ('nor',preprocessing.MinMaxScaler()),  # Data normalization
                            ('clf', OneVsRestClassifier(SVC()))])  # CLassification

            #list_c = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
            list_c = [1]
 
            #list_n = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550])
            list_n =[100]
            # Remember to add[2,\]2]
            best_score = 0
            epsilon = .005
            dictionary = {}

            for para_c in list_c:
                for para_n in list_n:
                    parameters = {'tfidf':[TfidfVectorizer(max_features=800, ngram_range=(1,4), norm='l2', encoding='latin-1', stop_words='english', analyzer='word')],
                                  'svd':[TruncatedSVD(n_components=para_n, n_iter=7, random_state=42)],
                                  'clf':[OneVsRestClassifier(SVC(kernel='linear', probability=True, C=para_c))]
                                 }
                    gs_clf = GridSearchCV(pipeline, parameters, cv=5, error_score='raise', scoring='f1')
                    gs_clf = gs_clf.fit(X_train, single_label)
                    current_score = gs_clf.best_score_
                    dictionary[current_score] = parameters

            for current_score in dictionary.keys():
                if current_score - epsilon > best_score:
                    best_score = current_score
                best_score_list.append(best_score)

            model_dict = dictionary[best_score]
            abs_filename_m = ett_h.generate_dynamic_path([base_folder_location, LabelType.THEME.value, model_folder_name, label+model_name]) 
            abs_filename_v = ett_h.generate_dynamic_path([base_folder_location, LabelType.THEME.value, vector_folder_name, label+vector_model_name]) 
            abs_filename_r = ett_h.generate_dynamic_path([base_folder_location, LabelType.THEME.value, dim_reductor_folder_name, label+dim_reductor_model_name]) 
            abs_filename_n = ett_h.generate_dynamic_path([base_folder_location, LabelType.THEME.value, normalizar_folder_name, label +normalizar_model_name]) 
            
            # Here to fit the training datasets to the  models with best score
            # vectorization
            vector = model_dict['tfidf'][0].fit(X_train, single_label)
            ett_h.save_model(vector, abs_filename_v)
            vectorized_df = vector.transform(X_train)
        
            # Balcancing
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(vectorized_df, single_label)
            
            # Feature selction
            svd = model_dict['svd'][0].fit(X_res,y_res)
            ett_h.save_model(svd, abs_filename_r)
            dim_reductor_df = svd.transform(X_res)
            
            # Normalizing
            min_max_scaler = preprocessing.MinMaxScaler()
            nor_model = min_max_scaler.fit(dim_reductor_df, y_res)
            ett_h.save_model(nor_model, abs_filename_n)
            scaled_df = nor_model.transform(dim_reductor_df)
            
            # Classifier
            clf = model_dict['clf'][0].fit(scaled_df, y_res)
            clf.fit(scaled_df, y_res)
            ett_h.save_model(clf, abs_filename_m)
        
        f1_score = json.dumps(best_score_list)   
        return "Training finished"
