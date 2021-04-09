#Web link :  https://casestudym5.herokuapp.com/index
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from joblib import load

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


# @app.route('/')
# def hello_world():
#     return 'Hello World!'

@app.route('/index', methods=['GET', 'POST'])  
def predict():
    if request.method == 'POST':
        file = request.files['file']
        path = "C:\\Users\\91998\\Desktop\\Git Commit\\HeroKuDeployment\\For_live\\IRIS\\uploads\\" + file.filename
        # file.save(os.path.join("/home/", file.filename))
        file.save(path)
        # print("file uploaded successfully")
        path = pd.read_csv(path)
        model = load("lightgbm")
        col = [ 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
        'wm_yr_wk', 'event_name_1', 'event_type_1',
        'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI',
        'sell_price', 'lag_28', 'lag_29', 'lag_30', 'rolling_mean_t7',
        'rolling_std_t7', 'year', 'month', 'day', 'week']
        start_time = time.time()
        for i in range(28, 31):
            index_name = "lag_"+str(i)
            path[index_name] = path.groupby(['id'])['demand'].transform(lambda x: x.shift(i))  
        path['rolling_mean_t7'] = path.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
        path['rolling_std_t7'] = path.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())    
        y_pred = model.predict(path[col])
        path['demand'] = y_pred
        predictions = path[['id', 'date', 'demand']]
        predictions = predictions.pivot_table( index = 'id', columns = 'date', values = 'demand')
        columns = ['F' + str(i + 1) for i in range(28)]
        predictions.columns = columns
       
        
        d =  str(time.time() - start_time) + str("seconds" )
        predictions['execution_time'] = d
        c = predictions.to_html(header="true",  table_id="table" )
        
        
    
        #https://stackoverflow.com/questions/39831894/json2html-python-json-data-not-converted-to-html#answer-39832966
        return (predictions.to_html(header="true",  table_id="table" ))
        # return  flask.render_template('index.html', message=[c,d])

    return  flask.render_template('index.html', message=' Upload file')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
