#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
 
"""     
This script is the entry point of running the UNISDR web application, which also configuring the database and JWT here.
""" 

__author__ = "Yihan Bao"
__copyright__ =  "Copyright 2019, United Nations OICT PSGD ETT"
__credits__ = ["Kevin Bradley", "Yihan Bao", "Praneeth Nooli"]
__date__ = "22 July 2019"
__version__ = "0.1"
__maintainer__ = "Yihan Bao"
__email__ = "yihan.bao@un.org"
__status__ = "Development"

import pandas as pd  # Package used to operate dataframe
import os  # Package used for accessing files
import logging  # Pakcage for logging info
import sys # Package used for saving logging info
from flask import Flask  # Pakcage for Flask API
from flask_restful import Api  # Pakcage for API
from flask import jsonify  # Package for jsonify object

# Save logging into local files
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='myLogs.log', filemode='w', format=logFormatter, level=logging.DEBUG)

# The main entry point for the application
app = Flask(__name__)
app.secret_key = 'abcdefghi'
api = Api(app)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message' : 'UNISDR Training FLASK API started'})

# After configuring all the api and app, import resources to link to endpointss
from controllers.trainer import hazardtrainer
from controllers.trainer import themetrainer

# Add a resource to the api
api.add_resource(hazardtrainer.TrainHazard, '/train/hazard')
api.add_resource(themetrainer.TrainTheme, '/train/theme')
