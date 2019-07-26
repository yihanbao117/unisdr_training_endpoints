#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Emerging Technologies Team Core Library for General Helper Functions
    
    Contains general helper based functions classes such as Enums and Exceptions
"""

__author__ = "Yihan Bao"
__copyright__ = "Copyright 2019, United Nations OICT PSGD ETT"
__credits__ = ["Kevin Bradley", "Yihan Bao", "Praneeth Nooli"]
__date__ = "27 February 2019"
__version__ = "0.1"
__maintainer__ = "Yihan Bao"
__email__ = "yihan.bao@un.org"
__status__ = "Development"

import pandas as pd  # DataFrame management
import logging  # Error Handling
import numpy as np  # Mathematical calcs
import pickle  # Used for loading models
import sklearn  # Used for model generation via pickle
from enum import Enum  # Used for custom Enums
from constants import Encoding  # Used to identify character encoding
import os # Pakcage used for reading and writing files

# Base Error class
class Error(Exception):


    """ETT Base class for other custom exceptions"""
    pass

# Custom Error Class
class InvalidLabelsToModelsError(Error):


   """The number of labels does not match the number of expected corresponding models"""
   pass

# Wrapper Class 
class Helper:

    ## 
    # This function is used to load a CSV file based on the 
    # filename and return this output.
    # @param filename The source file, as an string.
    # @return A DataFrame of the CSV file data.
    # @see OSError
    # @see Exception
    @staticmethod
    def load_csv(filename):

        try:
            return pd.read_csv(filename, encoding=Encoding.LATIN_1.value)
        except OSError:
            logging.error("OSError occurred", exc_info=True)
        except Exception:
            logging.error("Exception occurred", exc_info=True)

    ## 
    # This function is used to load a json file based on the 
    # filename and return this output.
    # @param filename The source file, as an string.
    # @return A DataFrame of the json file data.
    # @see OSError
    # @see Exception
    @staticmethod
    def load_json(filename):

        try:
            return pd.read_json(filename)
        except OSError:
            logging.error("OSError occurred", exc_info=True)
        except Exception:
            logging.error("Exception occurred", exc_info=True)
    
    ## 
    # This function is used to load a Model based on the 
    # filename and return this object.
    # @param filename The source file, as an string.
    # @return An Object which is the model.
    @staticmethod
    def load_model(filename):

        try:
            with open(filename, 'rb') as model_file:
                return pickle.load(model_file)
        except OSError:
            logging.error("OSError occurred", exc_info=True)
        except Exception:
            logging.error("Exception occurred", exc_info=True)
    
    #0419
    @staticmethod
    def save_model(model_name,filename):

        try:
            return pickle.dump(model_name,open(os.path.expanduser(filename),'wb'))
        except OSError:
            logging.error("OSError occurred", exc_info=True)
        except Exception:
            logging.error("Exception occurred", exc_info=True)
    
    ##
    # This function is used to load data based on the delimiter
    # @param filename String file path
    # @param delChar Character delimiter
    # @returns Tuple of values
    @staticmethod # Change the function  #  The outpurt is one list
    def load_data_common_separated(filename,delChar):

        try: 
            text_file = open(filename,'r')
            return text_file.read().split(delChar)
        except OSError:
            logging.error("OSError occurred", exc_info=True)
        except Exception:
            logging.error("Exception occurred", exc_info=True)
    
    ##
    # This function is used to create an empty DataFrame matching
    # the dimensions of the data expected to populate it
    # @param dataFrame DataFrame initial dataFrame
    # @param num Number of columns
    # @returns DataFrame with the correct dimensions 
    @staticmethod
    def provision_data_frame(dataFrame,num):

        try:
            provisioned_data = pd.DataFrame(np.zeros((len(dataFrame), num)))
            return provisioned_data
        except:
            if (dataFrame == None) | (num == None):
                logging.error("NameError occurred:",exc_info=True)
            else:
                logging.error("Exception occurred:",exc_info=True)
    
    ## 
    # Method used to create a DataFrame with the column names specified
    # @param dataFrame DataFrame of strings for each part of the path
    # @param colnames Tuple of strings with the column names
    # @returns DataFrame with the columns provisioned
    @staticmethod
    def provision_named_data_frame(dataFrame,colnames):

        try:
            provisioned_named_data = pd.DataFrame(dataFrame, columns=colnames)
            return provisioned_named_data
        except:
            if (dataFrame == None) | (colnames == None):
                logging.error("NameError occurred:",exc_info=True)
            else:
                logging.error("Exception occurred:",exc_info=True)

    ##
    # Method used to build a dynamic filepath
    # @param parts List of strings for each part of the path  
    # @returns String which is the combined file path
    @staticmethod
    def generate_dynamic_path(parts):

        try:
            if len(parts) > 1:
                return '/'.join(parts)
        except Exception:
                logging.error("Exception occurred",exc_info=True)

    ## TODO some form of verification to check parameters are not null or empty-DONE
    # Method used to string to dataframe
    # @param input_data A string
    # @param list_colname A list contains all the column names
    # @returns A dataframe with column name
    @staticmethod
    def string_to_dataframe(input_data,list_colname):

        try:
            input_data = pd.DataFrame(input_data,columns=list_colname)
            return input_data
        except:
            if (input_data == pd.DataFrame) | (list_colname == []):
                logging.error("NameError occurred:",exc_info=True)
            else:
                logging.error("Exception occurred:",exc_info=True)

    ## 
    # This function simply concatinates columns 
    # @param colnames Tuple of column names as Strings
    # @param data DataFrame of the original data to subset and concat
    # @returns concat_cols DataFrame of concatinated columns imto a single columne
    @staticmethod
    def concatinate_data_columns(colnames,data):
        
        try:
            concat_cols = data[colnames].apply(lambda x: ''.join(x), axis=1)
            return concat_cols
        except:
            if (colnames == None) | (data == None):
                logging.error("NameError occurred:",exc_info=True)
            else:
                logging.error("Exception occurred:",exc_info=True)


