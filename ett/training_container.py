#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Acts as a testing mechanism for the UNISDR solution.

    The UNISDR solution utilizes the ETT core modules which
    are a set of modules of common / generalized functions to use
    with the ETT Data Science Standards.
"""

import os  # Package used for accessing files
import time  # Calculcate time differences
import pandas as pd  # Dataframe operations
from helper import Helper as ett_h  # ETT Helper methods
from transformer import Transformer as ett_t  # ETT Transformer methods
from constants import JobType   # Enum for job type
from cleanser import Cleanser as ett_c
from sklearn.model_selection import train_test_split
from constants import RegexFilter        # Regex Filter Enums
from constants import Language                      # KV Language Enums 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

# Wrapper Class
class TrainContainer:


    # Class level variabbles used to populate the dependencies required to run the
    # UNISDR solution
    models = []
    data = pd.DataFrame()
    labels = []

    # Flag based variables
    num_of_labels = 0

    # File and path based variables
    base_folder_location = os.path.dirname(os.path.abspath(__file__))
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
    colnames = ["title", "textData", "label"]

    ##
    # Sets class variables and then sequentially loads the dependencies
    # and calls the classification code
    # @param job_type JobType identifies whether its batch or single
    # @param label_type ClassificationType identifies the classification type
    #def __init__(self, job_type = JobType.BATCH, label_type = ClassificationType.HAZARD):
    def __init__(self, label_type):
        #self.job_type = job_type
        self.label_type = label_type
        # Preload IO based data
        self.load_data()
        self.load_labels()
        self.data_cleann_transform()
        self.training_model()
        # Trigger on API call
        #self.perform_classification()

    # Method used to load the actual data to be classified
    # Populates the local data variable
    def load_data(self):
        abs_filename = ett_h.generate_dynamic_path([self.base_folder_location, self.label_type.lower(), self.data_file_name])
        input_data = ett_h.load_json(abs_filename)
        self.data_title = input_data['title']
        self.data_textData = input_data['textData']
        self.raw_label = input_data['label']
        self.data = ett_t.transform_data_to_dataframe_basic(input_data, self.colnames)
        self.label_df = ett_t.one_hot_encoding(self.raw_label)  #17 labels dataframe
    
    # Method used to load the list of labels
    # Populates the local labels list and counts the number of labels
    def load_labels(self):  # The output is a list
        abs_filename = ett_h.generate_dynamic_path([self.base_folder_location, self.label_type.lower(), self.label_file_name])
        self.label = (ett_h.load_data_common_separated(abs_filename, ','))
        self.num_of_labels = len(self.label)
        # Asign the label_df column name
        self.label_df.columns = self.label
        
    # Deal with input data
    def data_cleann_transform(self):
        # Removed all non alphanumeric characters
        nan_cleaned_data = ett_c.clean_dataframe_by_regex(self.data, RegexFilter.NON_ALPHA_NUMERIC.value) 
        # Removed all digits
        d_cleaned_data = ett_c.clean_dataframe_by_regex(nan_cleaned_data, RegexFilter.DIGITS_ONLY.value)
        # Remove non-English text
        l_cleaned_data = ett_c.remove_non_iso_words(d_cleaned_data, Language.ENGLISH.value)  
        # Remove English stop words
        rew_cleaned_data = ett_c.remove_language_stopwords(l_cleaned_data, Language.ENGLISH.name) 
        # Transform text to lowercase
        l_transformed_data = ett_t.lowercase(rew_cleaned_data)
        # Transform text to core words i.e. playing > play
        le_transformed_data = ett_t.stemming_mp(l_transformed_data)
        # Return the newly transformed data
        self.data = le_transformed_data

    # Method which acts as the builder #Automatically select 
    def training_model(self):
        # Split the data into 0.8 training datasets and 0.2 testing datasets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.label_df, test_size=0.2)
        for i in range(self.num_of_labels):
            self.single_label = self.y_train.iloc[:,i]
            label = self.label[i]
            print(self.X_train.shape)
            print(self.single_label.shape)
            pipeline = imbPipeline([
                            ('tfidf', TfidfVectorizer()),
                            ('oversample', SMOTE(random_state=42)),
                            ('svd', TruncatedSVD()),
                            ('nor',preprocessing.MinMaxScaler()),
                            ('clf', OneVsRestClassifier(SVC()))])
            #list_c = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
            list_c = [1]
            #list_c =  np.linspace(0,1,5)
            list_n =[100]
            # Remember to add[2,\]2]
            best_score = 0
            epsilon = .005
            dictionary = {}
            for para_c in list_c:
                for para_n in list_n:
                    parameters = {'tfidf':[TfidfVectorizer(max_features = 800, ngram_range=(1,4), norm='l2', encoding='latin-1', stop_words='english', analyzer='word')],
                                'svd':[TruncatedSVD(n_components=para_n, n_iter=7, random_state=42)],
                                'clf':[OneVsRestClassifier(SVC(kernel='linear', probability=True, C=para_c))]
                                }
                    gs_clf = GridSearchCV(pipeline, parameters, cv=5, error_score='raise', n_jobs = -1)
                    gs_clf = gs_clf.fit(self.X_train, self.single_label)
                    current_score = gs_clf.best_score_
                    dictionary[current_score] = parameters
            for current_score in dictionary.keys():
                if current_score - epsilon > best_score:
                    best_score = current_score
                    print("this is best score",best_score)
            model_dict = dictionary[best_score]
            abs_filename_m = ett_h.generate_dynamic_path([self.base_folder_location, self.label_type.lower(), self.model_folder_name, label + self.model_name]) 
            abs_filename_v = ett_h.generate_dynamic_path([self.base_folder_location, self.label_type.lower(), self.vector_folder_name, label + self.vector_model_name]) 
            abs_filename_r = ett_h.generate_dynamic_path([self.base_folder_location, self.label_type.lower(), self.dim_reductor_folder_name, label + self.dim_reductor_model_name]) 
            abs_filename_n = ett_h.generate_dynamic_path([self.base_folder_location, self.label_type.lower(), self.normalizar_folder_name, label + self.normalizar_model_name]) 
            # Here to fit the training datasets to the  models with best score
            # vectorization
            vector = model_dict['tfidf'][0].fit(self.X_train, self.single_label)
            ett_h.save_model(vector, abs_filename_v)
            vectorized_df = vector.transform(self.X_train)
            # Balcancing
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(vectorized_df, self.single_label)
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
TrainContainer('THEME')
