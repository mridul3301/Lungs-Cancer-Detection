from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.views.generic import TemplateView
import pandas as pd
import numpy as np
import joblib

model = joblib.load("./nof_svm.pkl")
pipeline = joblib.load("./nof_popeline.pkl")

def homePageView(request):
    return render(request, 'index.html')

def output(request):
    data = {}
    output = None
    
    AGE = request.POST.get('age')
    GENDER = request.POST.get('gender')
    SMOKING = request.POST.get('smoker')
    YELLOW_FINGERS = request.POST.get('fingers')
    ANXIETY = request.POST.get('anxiety')
    PEER_PRESSURE = request.POST.get('peer')
    CHRONIC_DISEASE = request.POST.get('chronic')
    FATIGUE = request.POST.get('fatigue')
    ALLERGY = request.POST.get('allergy')
    WHEEZING = request.POST.get('wheezing')
    ALCOHOL_CONSUMING = request.POST.get('alcohol')
    COUGHING = request.POST.get('coughing')
    SHORTNESS_OF_BREATH = request.POST.get('breath')
    SWALLOWING_DIFFICULTY = request.POST.get('swallow')
    CHEST_PAIN = request.POST.get('chest')
        
    data = {
            'GENDER':GENDER,
            'AGE':AGE,
            'SMOKING':SMOKING,
            'YELLOW_FINGERS':YELLOW_FINGERS,
            'ANXIETY':ANXIETY,
            'PEER_PRESSURE':PEER_PRESSURE,
            'CHRONIC_DISEASE':CHRONIC_DISEASE,
            'FATIGUE ':FATIGUE,
            'ALLERGY ':ALLERGY,
            'WHEEZING':WHEEZING,
            'ALCOHOL_CONSUMING':ALCOHOL_CONSUMING,
            'COUGHING':COUGHING,
            'SHORTNESS_OF_BREATH':SHORTNESS_OF_BREATH,
            'SWALLOWING_DIFFICULTY':SWALLOWING_DIFFICULTY,
            'CHEST_PAIN':CHEST_PAIN
        }
    df = pd.DataFrame(data, index=[0])
    df['AGE'] = pipeline.transform(df['AGE'].values.reshape(-1, 1))
    output = model.predict(df)
    
    if output[0] == 0:
        data['output'] = "You are safe from Lung Cancer"    
    else:
        data['output'] = "You are at risk of Lung Cancer"

    
    return render(request, 'output.html', data)



