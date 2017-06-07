from flask import Flask, render_template, request, jsonify
from data import Articles
import pandas as pd
import numpy as np
import re
from XGBoostModel import *


#!!! pip install Flask-PyMongo
from flask_pymongo import PyMongo
from pymongo import MongoClient
import urllib.request, json

app = Flask(__name__)
mongo = PyMongo(app)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/register', methods=['POST'])
def register():
    return render_template('home.html')

@app.route('/flag', methods=['POST'])
def flag():
    return render_template('flag.html')


@app.route('/history', methods=['GET'])
def history():
    #add pagination
    # start = request.args.get('start', default=0, type=None)
    # limit = request.args.get('limit', default=10, type=None)
    client = MongoClient()
    #set database that collection is in
    db = client.fraud
    df = pd.DataFrame(list(db.review.find()))

    select_columns = ['object_id', 'fraud_state', 'proba', 'name', 'org_name', 'venue_name', 'venue_state', 'venue_country']
    df = df[select_columns]
    data_temp=df.to_html()
    #re.sub('<[^>]*>', '', data_temp)
    data = df.to_html()

    return render_template("history.html",data=data)

@app.route('/review')
def review():
    with urllib.request.urlopen("http://galvanize-case-study-on-fraud.herokuapp.com/data_point") as url:
        #data = json.loads(url.read().decode())
        temp = url.read().decode()
        data = json.loads(temp)

    client = MongoClient()
    #set database that collection is in
    db = client.fraud
    #add json call to mongo
    db.review.update(data, data, upsert=True)

    # --- fix this with real model
    current_proba = np.random.uniform(0, 1)
    current_fraud = np.random.uniform(0, 1)

    db.review.update(data, data, upsert=True)

    db.review.update_one({'object_id' : data['object_id']},
         {'$set' : {'proba': current_proba}})

    db.review.update_one({'object_id' : data['object_id']},
         {'$set' : {'text_proba': current_fraud}})
    # end fix

    df = pd.DataFrame(list(db.review.find({'object_id' : data['object_id']})))

    ### add model here
    # ==================================
    model = XGBoostModel(use_rfc=True)
    model.unpickle()
    X, oid = model.load_one(temp)
    #mf = model.df
    proba = model.predict_proba(X)

    #mf = pd.read_csv('sample.csv')
    # ==================================
    ### add model here



    object_id = data['object_id']

    t_low,t_med,t_high = .25,.50,.75
    df['proba'] = proba
    fraud_state = 'low'
    if ((proba > t_low) and (proba <= t_med)):
        fraud_state = 'MED!'
    elif proba >= t_high :
        fraud_state = 'HIGH!'

    df['fraud_state'] = fraud_state

    db.review.update_one({'object_id' : data['object_id']},
         {'$set' : {'fraud_state': fraud_state}})



    return render_template("review.html",event_id=object_id, data=df.iloc[0,:].to_frame().to_html())

@app.route('/about')
def articles():
    return render_template('about.html', articles = articles)

if __name__ == '__main__':
    app.run(debug=True)
