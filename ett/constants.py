#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Emerging Technologies Team Core Library housing ENUM lists
    
    Contains functions that focus on Enum
"""


from enum import Enum    # Used for custom Enums

# List of encoding types
class Encoding(Enum):
    LATIN_1 = "latin1"

# List of job processing types
class JobType(Enum):
    BATCH = "BATCH"                                             # No comma between two constants otherwise the output will be tuple not value
    SINGLE = "SINGLE"

# List of regular expressions
class RegexFilter(Enum):
    NON_ALPHA_NUMERIC = "[^\w\s]"                               # ^ NOT, \w WORDS, \s WHITESPACE
    DIGITS_ONLY = "[\d]"                                        # \d DIGITS
    SQUARE_BRACKET = "\\[|\\]"                                  # | AND, \[ left squre bracket,\] right square bracket
    SINGLE_QUOTATION = "\\'"                                    # \' QUOTATION 
    SQUARE_BRACKET_AND_SINGLE_QUOTATION = "\\[|\\]|\\'"         # | AND, \[ left squre bracket,\] right square bracket
        
# KV list of languages and ISO code
class Language(Enum):
    ENGLISH = "en_US"
