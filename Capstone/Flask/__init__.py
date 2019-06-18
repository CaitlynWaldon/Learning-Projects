import flask
import dill 
import numpy as np
import pandas as pd 

app = flask.Flask(__name__)

with open('churner_model.pkl', 'rb') as f:
    PREDICTOR = dill.load(f)
##################################
@app.route("/")
def hello():
    return '''
    <body>
    <h2> Hello World! <h2>
    </body>
    '''

##################################
@app.route('/greet/<name>')
def greet(name):
    '''Say hello to your first parameter'''
    return "Hello, %s!" %name

@app.route('/predict', methods=["GET"])
def predict():
    city = flask.request.args['city']
    registered_via= flask.request.args['registered_via']
    payment_method_id = flask.request.args['payment_method_id']
    payment_plan_days = flask.request.args['payment_plan_days']
    actual_amount_paid = flask.request.args['actual_amount_paid']
    is_auto_renew = flask.request.args['is_auto_renew']
    is_cancel= flask.request.args['is_cancel']
    num_unq = flask.request.args['num_unq']
    total_songs = flask.request.args['total_songs']
    songs_repeated = flask.request.args['songs_repeated']
    membership_length = flask.request.args['membership_length']

    item = pd.DataFrame([[city, registered_via, payment_method_id, payment_plan_days, actual_amount_paid,is_auto_renew,is_cancel,num_unq,total_songs,songs_repeated,membership_length]], columns=['city', 'registered_via', 'payment_method_id', 'payment_plan_days','actual_amount_paid', 'is_auto_renew', 'is_cancel', 'num_unq','total_songs', 'songs_repeated', 'membership_length'])
    
    #item = np.array([pclass, sex, age, fare, sibsp])
    print (item)
    score = PREDICTOR.predict_proba(item)

    results = {'churn chances': score[0,1], 'non curn chances': score[0,0]}
    return flask.jsonify(results)

##################################
#@app.route('/page')
#def show_page():
#    return flask.render_template('dataentrypage.html')

##################################
@app.route('/page', methods=['POST', 'GET'])
def page():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

       inputs = flask.request.form

       city = inputs['city'][0]
       registered_via = inputs['registered_via'][0]
       payment_method_id = inputs['payment_method_id'][0]
       payment_plan_days = inputs['payment_plan_days'][0]
       actual_amount_paid = inputs['actual_amount_paid'][0]
       is_auto_renew = inputs['is_auto_renew'][0]
       is_cancel = inputs['is_cancel'][0]
       num_unq = inputs['num_unq'][0]
       total_songs = inputs['total_songs'][0]
       songs_repeated = inputs['songs_repeated'][0]
       membership_length = inputs['membership_length'][0]

       item = pd.DataFrame([[city, registered_via, payment_method_id, payment_plan_days, actual_amount_paid,is_auto_renew,is_cancel,num_unq,total_songs,songs_repeated,membership_length]], columns=['city', 'registered_via', 'payment_method_id', 'payment_plan_days','actual_amount_paid', 'is_auto_renew', 'is_cancel', 'num_unq','total_songs', 'songs_repeated', 'membership_length'])
       print (item)
       score = PREDICTOR.predict_proba(item)
       #results = {'survival chances': score[0,1], 'death chances': score[0,0]}
       churn = int(score[0,1] * 100)
       nonchurn = int(score[0,0] * 100)
    else:
        churn = 0
        nonchurn = 0
    return flask.render_template('dataentrypage.html', churn=churn, nonchurn=nonchurn)

##################################
if __name__ == '__main__':
    app.run(debug=True)