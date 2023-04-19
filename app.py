import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import _base
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from sklearn.svm import SVC
import pickle

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    my_df = pd.read_csv('lung_cancer_dataset.csv')

    #features = [x for x in request.form.values()]
    
    age = int(request.form['age'])
    smoker = int(request.form['Smoker'])
    yf = int(request.form['Yellow Fingers'])
    anxiety = int(request.form['Anxiety'])
    pp = int(request.form['Peer Pressure'])
    cd = int(request.form['Chronic Disease'])
    fatigue = int(request.form['Fatigue'])
    allergy = int(request.form['Allergy'])
    wheezing = int(request.form['Wheezing'])
    alcohol = float(request.form['Alcohol'])
    cough = int(request.form['Coughing'])
    bs = int(request.form['Breath Shortness'])
    ds = int(request.form['Difficulty'])
    cp = float(request.form['Chest Pain'])
    gender = int(request.form['Gender'])

    #X_target = features[1]
    #y = [1]
    scaler=StandardScaler()
    my_df['AGE'] = scaler.fit_transform(my_df['AGE'].values.reshape(-1, 1))
    Age = scaler.transform([[age]])

    data = {'Age': [Age],
            'Smoker': [smoker],
            'Yellow Fingers': [yf],
            'Anxiety': [anxiety],
            'Peer Pressure': [pp],
            'Chronic Disease': [cd],
            'Fatigue': [fatigue],
            'Allergy': [allergy],
            'Wheezing': [wheezing],
            'Alcohol': [alcohol],
            'Coughing': [cough],
            'Breath Shortness': [bs],
            'Difficulty': [ds],
            'Chest Pain': [cp],
            'Gender': [gender],
            }

    df = pd.DataFrame(data)

    model = joblib.load("SVC.pkl")

    y = model.predict(df.values)

    if (y == 1):
        op = 'Positive'
    else:
        op = 'Negative'
    #dti_model = models.model_pretrained(path_dir = 'DTI_Model')
    #y_pred = dti_model.predict(X_pred)

    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #output = str(y_pred)

    return render_template('index.html', prediction_text='Lung Cancer = {}'.format(op))

@app.route('/predict_api',methods=['GET', 'POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)