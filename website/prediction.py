from flask import Blueprint, render_template, request, redirect, url_for, session, json
import pandas as pd 
from . import model_pred
from .data_access.resultadoDA import ResultadoDataAccess
from .data_access.condicionDA import CondicionDataAccess
from .data_access.visitDA import VisitDataAccess
import matplotlib.pyplot as plt
import os
import seaborn as sns


prediction = Blueprint('prediction', __name__)

#Init DA classess
resultadoDA = ResultadoDataAccess()
condicionDA = CondicionDataAccess()
visitDA = VisitDataAccess()

#Method to save the feature_importance table from the model
def save_importance():
    importance_df = model_pred.feature_importance()
    importance_df.sort_values(by='importance', ascending=False)
    # Create a vertical bar plot using seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Features', y='importance', data=importance_df, palette='viridis')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance (Vertical Bar Plot)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    # Saves the figure in the static directory  
    plt.savefig(os.path.join('website', 'static', 'images', 'feature_importance.png')) 
    plt.close()
    
#Method to construct a pandas DataFrame to feed the model
def construct_df(edad, sexo, angina, rbp, chol, fbs, rest_ecg, max_hr, exang, oldpeak, slope):

    column_headers = [
        "Edad", "Sexo", "Angina", "PresionArterialDescanso", "Colesterol",
        "NivelesAzucarAyuna", "ECGDescanso", "MaxRitmoCardiaco", "AnginaEjercicio",
        "Oldpeak(SegmentoST)", "Slope(SegmentoST)"
    ]

    # Create an empty DataFrame with the defined column headers
    df = pd.DataFrame(columns=column_headers)

    # Create a dictionary with values for the row
    row_data = {
        "Edad": [edad],
        "Sexo": [sexo],
        "Angina": [angina],
        "PresionArterialDescanso": [rbp],
        "Colesterol": [chol],
        "NivelesAzucarAyuna": [fbs],
        "ECGDescanso": [rest_ecg],
        "MaxRitmoCardiaco": [max_hr],
        "AnginaEjercicio": [exang],
        "Oldpeak(SegmentoST)": [oldpeak],
        "Slope(SegmentoST)": [slope]
    }

    # Create a DataFrame from the row_data dictionary and its headers
    df = pd.DataFrame(row_data, columns=column_headers)
    return df

#Method to process condition ids
#Returns the same tuple with values instead of ids
def process_conditionIds(conditionsIds):
    #list comprehension, returns a list with new updated values
    updated_conditions = []
    for _, value in conditionsIds.items():
        updated_conditions.append(condicionDA.get_condicion_by_id(value).get_cantidad())
    return updated_conditions

#Method to predict using model
def __predict(conditionsIds, patient_details):
    #Process ids and give values to each
    processed_parameters = process_conditionIds(conditionsIds)

    #patient_details, contains (age, sex, visitId) of the patient
    age = patient_details["age"]
    sex = patient_details["sex"]

    #Convert sex to integer, 1 = male, 0 = female
    if sex == 'M':
        sex = 1
    else:
        sex = 0

    #Add patient details to the parameters
    processed_parameters.insert(0, sex)
    processed_parameters.insert(0, age)

    #Organize into a pd.DataFrame
    test = construct_df(*processed_parameters)

    #Predict method from PyTorchTabular
    result = model_pred.predict(test)

    return result

####################
# Prediction views #
####################     

#View route for model prediction
@prediction.route('/predict', methods=['GET'])
def predict():
    if 'loggedin' in session:
        #Receive values to predict
        conditionsIds = request.args.get('conditions')
        patient_details = request.args.get('patient_details')

        #Deserialize json received
        details = json.loads(patient_details)
        conditions = json.loads(conditionsIds)

        #Store visitId
        visitId = details["visitId"]

        #Get the resulting prediction value
        result = __predict(conditions, details)
        value = result.iloc[0]['prediction']

        #Produce a prediction and store in a RESULTADO instance
        resultadoDA.store_resultado(value, visitId)

        #Get patientId
        patientId = visitDA.get_patient_id_by_visit(visitId)
        
        return redirect(url_for('patient.patient_record', patientId=patientId))
    return redirect(url_for('views.main'))

#View route for the prediction result analysis
@prediction.route('/predictive_analysis', methods=['GET'])
def predictive_analysis():
    if 'loggedin' in session:
        visitId = request.args.get('visitId')
        #Saves the feature importance table
        save_importance()
        return render_template('result_analysis.html', visitId=visitId)
    return redirect(url_for('views.main'))