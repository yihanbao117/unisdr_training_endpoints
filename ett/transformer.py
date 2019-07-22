#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Emerging Technologies Team Core Library for Transformation Functions

    Contains methods that are used to transform data in some way
"""


import nltk                                   # Package used to do text analysis 
import logging                                # Package used to handling errors
import pandas as pd                           # Operation dataframe
from constants import JobType                 # Enum for job type
from helper import Helper as ett_h            # Core package of ETT 
from textblob import Word                     # Package used to do lemmatization
#nltk.data.path.append("/Users/kevin/Desktop/unisdr/nltk_data")
#from pathos.multiprocessing import ProcessingPool as Pool 
from multiprocessing import Pool              # Package used for multiprocessing
from nltk import WordNetLemmatizer            # Text data normalization-lemmatization
from nltk.stem import PorterStemmer           # Text data normalization-stemming
from sklearn import preprocessing             # Use for normalize the datasets to [0,1]
from sklearn.preprocessing import MultiLabelBinarizer 

class Transformer:

    # concatinate_data_columns move TO HELPER 

    ## 
    # This function simply transforms text to lowercase           
    # @param text String input text
    # @returns String lowercased text
    # Here the text actually is a dataframe with text data
    # Check if the dataframe is empty or not
    @staticmethod
    def lowercase(text):
        # pd.DataFrame is an empty dataframe
        if text == pd.DataFrame:
            print("Your text data is empty, please check your input data")
        else:
            text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))  
            return text

    ## 
    # This function simply transforms text to lowercase           
    # @param text String input text
    # @returns String stemmed text
    # Here the text actually is a dataframe with text data
    # Check if the dataframe is empty or not
    @staticmethod
    def stemming(text):
        try:
            porter_stemmer = PorterStemmer()
            text = text.apply(lambda x: " ".join([porter_stemmer.stem(word) for word in x.split()]))
            return text
        except Exception:
            logging.error("Here occurred",exc_info=True)

    ## 
    # This function simply transforms text to lowercase           
    # @param text String input text
    # @returns String stemmed text
    # Here the text actually is a dataframe with text data
    # Check if the dataframe is empty or not
    @staticmethod
    def stemming_mp(text,cores=4):   
       # pd.DataFrame is an empty dataframe
        if text == pd.DataFrame:
            print("Your text data is empty, please check your input data")
        else:
            with Pool(processes=cores) as pool:
                #l_st = time.time()
                porter_stemmer = PorterStemmer()
                text = text.apply(lambda x: " ".join([porter_stemmer.stem(word) for word in x.split()]))
                #l_et = time.time()
                #print("Time of lemmatization: " , str(l_st-l_et))
            return text

    ##            
    # This function simply transforms text morphologically by removing inflectional endings
    # for instance cats > cat                                          
    # @param text String input text                                    
    # @returns String rooted text
    @staticmethod                                                   
    def lemmatization(text):
        text = text.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        return text
    
    # Here the text actually is a dataframe with text data
    # Check if the dataframe is empty or not
    @staticmethod
    def lemmatization_mp(text,cores=2):
        # pd.DataFrame is an empty dataframe
        if text == pd.DataFrame:
            print("Your text data is empty, please check your input data")
        else:
            with Pool(processes=cores) as pool:
                #l_st = time.time()
                wlemm = WordNetLemmatizer()
                result = pool.map(wlemm.lemmatize, text)
                #l_et = time.time()
                #print("Time of lemmatization: " , str(l_st-l_et))
            return result
    
    ## 
    # This function simply calls the transform method of the model object
    # Method using these calculated parameters apply the transform 8ij ation to a particular dataset
    # @param model Object representing a model
    # @param dataFrame DataFrame to be used to transform
    # @returns concat_cols DataFrame of concatinated columns imto a single column   
    # @see sklearn.transform()
    @staticmethod
    def perform_model_transformation(model, dataFrame): 
        try:
            return model.transform(dataFrame)
        except AttributeError:
            logging.error("AttributeError occurred: ",exc_info=True)
        except Exception:
            logging.error("Exception occurred",exc_info=True)

    ## 
    # Method used to transform the input data to a dataframe
    # @param input_data DataFrame the initial data loaded into the app 
    # @returns DataFrame
   
    # If want to call the functions within the same class, need ad another argument: self
    # But then this self argumnet will be used by other function(which will cause unnecessary)
    # So moving the string_to_dataframe and concatinate_data_columns to another file(helper.py)
    @staticmethod
    def transform_data_to_dataframe(job_type, input_data, list_column):   
        # SINGLE                                                        
        if job_type == JobType.BATCH.value:                             
            dataframe_single = ett_h.string_to_dataframe(input_data,list_column)
            combined_data = ett_h.concatinate_data_columns(list_column, dataframe_single)
        # BATCH
        else:  
            combined_data = ett_h.concatinate_data_columns(list_column, input_data)
        return combined_data

    ## 
    # Method used to transform the input data to a dataframe
    # @param input_data DataFrame the initial data loaded into the app 
    # @returns DataFrame
    @staticmethod
    def transform_data_to_dataframe_basic(input_data, list_column):   #0419
        combined_data = ett_h.concatinate_data_columns(list_column, input_data)
        return combined_data
    ##
    # Method used to combined all dataframe in a list to one big dataframe based on x or y
    # @param list_dataframe A list contains all dataframe that you want to combine together
    # @param axis_num 1 or 0,axis_num = 1 for x; axis_num = 1 for y
    # @returns Combined dataframe
    @staticmethod
    def combine_dataframe(list_dataframe, axis_num): 
        for i in list_dataframe:
             result_model = pd.concat([i for i in list_dataframe],axis = axis_num)
        return result_model
    
    @staticmethod 
    def one_hot_encoding(dataframe):
        label_new = []
        for i in dataframe:
            add_space = " "+i
            label_new.append(add_space)
        label_col = pd.Series(label_new)

        label_list = []
        for line in label_col:
            line = line.split(',')
            label_list.append(line)

        label_encode = MultiLabelBinarizer()
        label_array = label_encode.fit_transform(label_list)
        encoded_label = pd.DataFrame(label_array)
        return encoded_label