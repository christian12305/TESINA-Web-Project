from flask import Blueprint, render_template, request, redirect, url_for, session, json
import pandas as pd 
from . import model_pred
from .dataAccess.resultadoDA import ResultadoDataAccess
from .dataAccess.condicionDA import CondicionDataAccess
from .dataAccess.visitDA import VisitDataAccess

prediction = Blueprint('prediction', __name__)

#Init DA classess
resultadoDA = ResultadoDataAccess()
condicionDA = CondicionDataAccess()
visitDA = VisitDataAccess()

#View route for the prediction result analysis
@prediction.route('/predictive_analysis', methods=['GET', 'POST'])
def predictive_analysis():
    if 'loggedin' in session:
        if request.method == 'POST':
            pass
        visitId = request.args.get('visitId')
        return render_template('result_analysis.html', visitId=visitId)
    
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

#Process condition ids
#Returns the same tuple with values instead of ids
def process_conditionIds(conditionsIds):
    #list comprehension, returns a list with new updated values
    updated_conditions = []
    for key, value in conditionsIds.items():
        updated_conditions.append(condicionDA.get_condicion_by_id(value).get_cantidad())
    #updated_conditions = {key: condicionDA.get_condicion_by_id(value).get_cantidad() for key, value in conditionsIds.items()}
    return updated_conditions
    
#Method to predict using model
def __predict(conditionsIds, patient_details):
    #Process ids and give values to each
    processed_parameters = process_conditionIds(conditionsIds)

    #patient_details, contains (age, sex, visitId) of the patient
    age = patient_details["age"]
    sex = patient_details["sex"]

    #Convert sex to integer
    if sex == 'M':
        sex = 1
    else:
        sex = 0

    #Add patient details to the parameters
    processed_parameters.insert(0, sex)
    processed_parameters.insert(0, age)

    #Organize into a pd.DataFrame
    test = construct_df(*processed_parameters)

    #Predict
    #pred_model._tab_model
    result = model_pred.pred(test)

    return result
        

#View route for model prediction
@prediction.route('/predict', methods=['GET'])
def predict():
    if 'loggedin' in session:
        #Receive values to predict for
        conditionsIds = request.args.get('conditions')
        patient_details = request.args.get('patient_details')
        details = json.loads(patient_details)
        conditions = json.loads(conditionsIds)

        visitId = details["visitId"]

        #Get the resulting prediction value
        result = __predict(conditions, details)
        value = result.iloc[0]['prediction']

        #Produce a prediction and store in a RESULTADO instance
        resultadoDA.store_resultado(value, visitId)

        #Get patientId
        patientId = visitDA.get_patient_id_by_visit(visitId)

        return redirect(url_for('patient.patient_record', patientId=patientId))
    return redirect(url_for('auth.login'))