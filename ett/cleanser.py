#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Emerging Technologies Team Core Library for Data Cleansing Functions
    
    Contains functions that focus on cleaning data
"""

__author__ = "Yihan Bao"
__copyright__ = "Copyright 2019, United Nations OICT PSGD ETT"
__credits__ = ["Kevin Bradley", "Yihan Bao", "Praneeth Nooli"]
__date__ = "27 February 2019"
__version__ = "0.1"
__maintainer__ = "Yihan Bao"
__email__ = "yihan.bao@un.org"
__status__ = "Development"

import enchant  # Package used to detect English words
import nltk  # Package used to do text analysis 
import re  # Package used to as replace function
import logging  # Package used to handling errors
import pandas as pd  # Package used to operate dataframe
from nltk.corpus import stopwords  # Corpus used to remove English stopwords
#nltk.data.path.append("/Users/kevin/Desktop/unisdr/nltk_data")

# Wrapper Class
class Cleanser:


    ##
    # This function is used to remove certain characters
    # from the DataFrame by using regular expressions
    # @param dataFrame DataFrame to be cleansed
    # @param regex String which represents the regex to find and replace with ''
    # Most frequenltli used "[^\w\s]" and "[\d]"
    # @return DataFrame of the cleansed data
    @staticmethod
    def clean_dataframe_by_regex(dataFrame,regex):
        try:
            dataFrame = dataFrame.apply(lambda x: re.sub(str(regex), "",x))  
            return dataFrame
        except TypeError:
            logging.error("TypeError occurred",exc_info=True)
        except Exception:
            logging.error("Exception occurred",exc_info=True)
    
    ## 
    # This function is used to remove non specific ISO words
    # from the DataFrame. For example non American English words
    # @param dataFrame DataFrame to be cleansed
    # @param iso String which represents the ISO code for the language.
    # @return DataFrame of the cleansed data
    # @see enchant.Dict() function used to construct a dictionary based on the iso code
    @staticmethod
    def remove_non_iso_words(dataFrame,iso):
        try:
            iso_dict = enchant.Dict(iso)
            dataFrame = dataFrame.apply(lambda x: " ".join(x for x in x.split() if iso_dict.check(x)))
            return dataFrame
        except TypeError:  
            logging.error("TypeError occurred",exc_info=True)
        except Exception:
            logging.error("Exception occurred",exc_info=True)
    
    ##
    # This function is used to remove language specific stopwords
    # from the DataFrame. For example the English word 'the'
    # @param dataFrame DataFrame to be cleansed
    # @param language String which represents the language
    # @return DataFrame of the cleansed data
    @staticmethod
    def remove_language_stopwords(dataFrame,language):
        try:
            stops = set(stopwords.words(language))
            sw_removed = dataFrame.apply(lambda x: " ".join(x for x in x.split() if x not in stops))
            return sw_removed
        except TypeError:
            logging.error("TypeError occurred",exc_info=True)
        except Exception:
            logging.error("Exception occurred",exc_info=True)

