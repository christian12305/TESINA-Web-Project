from flask import Blueprint, render_template, request, flash, redirect, url_for, session, json
from .dataAccess.patientDA import PatientDataAccess
from .dataAccess.visitDA import VisitDataAccess
from .dataAccess.condicionDA import CondicionDataAccess
from .business_logic.models import CondicionType

patient = Blueprint('patient', __name__)

#Init DA classess
patientDA = PatientDataAccess()
visitDA = VisitDataAccess()
condicionDA = CondicionDataAccess()


#################
# Patient views #
#################

#View route for the search patient endpoint with get and post methods
@patient.route('/search', methods=['GET', 'POST'])
def search_patient():
    #Verify if user is loggedin
    if 'loggedin' in session:
        if request.method == 'POST':
            #Received a search request
            input = request.form.get('search')
            #To find records starting with the data entered by the user
            input = input.strip()
            patients = patientDA.getPatients(input)
            return render_template("search_patient.html", patients=patients)
        return render_template("search_patient.html")
    return redirect(url_for('views.main'))

#View route for the create patient endpoint with get and post methods
@patient.route('/create', methods=['GET', 'POST'])
def create_patient():
    #Verify if user is loggedin
    if 'loggedin' in session:
        if request.method == 'POST':
            #Patient has been submitted for creation
            firstName = request.form.get('firstName')
            initial = request.form.get('initial')
            firstLastName = request.form.get('firstLastName')
            secondLastName = request.form.get('secondLastName')
            gender = request.form.get('gender')
            weight = request.form.get('weight')
            condition = request.form.get('condition')
            email = request.form.get('email')
            celullar = request.form.get('cel')
            birthDate = request.form.get('birthDate')

            #Search if patient email already exists
            patient = patientDA.get_patient_by_email(email)

            #Validations
            if patient:
                flash('Email already exists.', category='e')
            elif len(email) < 5:
                flash('Email must be greater than 3 characters.', category='e')
            elif len(firstName) < 2:
                flash('First name must be greater than 1 character.', category='e')
            elif len(initial) > 1:
                flash('Initial must contain only 1 character.', category='e')
            elif len(firstLastName) < 2:
                flash('Last name must be greater than 1 character.', category='e')
            else:

                #Create a patient with the parameters sent
                params = (firstName, initial, firstLastName, secondLastName, birthDate, gender, weight, condition, email, celullar)
                patientId = patientDA.store_patient(*params)
                record = patientDA.get_patient_record(patientId)
                recordId = record.get_id()
                #Notify the user
                flash('Patient created!', category='s')
                #Send to the new visit
                return redirect(url_for('patient.new_visit', patientId=patientId, recordId=recordId))
        #GET method
        return render_template("create_patient.html")
    return redirect(url_for('views.main'))

#View route for the patient record endpoint
@patient.route('/patient_record', methods=['GET'])
def patient_record():
    #Verify if user is loggedin
    if 'loggedin' in session:
        patientId = request.args.get('patientId')
        patient = patientDA.get_patient_by_id(patientId)
        visits = visitDA.get_patient_visits(patientId)
        return render_template('patient_record.html', patient=patient, visits=visits, patientId=patientId)
    return redirect(url_for('views.main'))
 
#View route for the new visit endpoint
@patient.route('/new_visit', methods=['GET', 'POST'])
def new_visit():
    if 'loggedin' in session:
        #Request to load page
        if request.method == 'GET':
            recordId = request.args.get('recordId')
            patientId = request.args.get('patientId')
            return render_template('new_visit.html', recordId=recordId, patientId=patientId)

        #Obtain all form inputs

        #Hidden inputs
        patientId = request.form['patientId']
        recordId = request.form['recordId']
        #User inputs
        cp = request.form['chest_pain']
        rbp = request.form['resting_bp']
        chol = request.form['cholesterol']
        fbs = request.form['fasting_sugar']
        rest_ecg = request.form['rest_ecg']
        max_hr = request.form['max_heart_rate']
        exang = request.form['exang']
        vessels = request.form['major_vessels']
        thal = request.form['thal']
        slope = request.form['slope']
        oldpeak = request.form['oldpeak']
        

        #Create a VISITA instance
        visit = visitDA.new_visit(recordId)
        visitId = visit.get_id()

        #Create a instance of table CONDICION and
        condicionId1 = condicionDA.store_condicion(CondicionType.ChestPain, cp)
        condicionId2 = condicionDA.store_condicion(CondicionType.RBP, rbp)
        condicionId3 = condicionDA.store_condicion(CondicionType.Chol, chol)
        condicionId4 = condicionDA.store_condicion(CondicionType.FBS, fbs)
        condicionId5 = condicionDA.store_condicion(CondicionType.RestECG, rest_ecg)
        condicionId6 = condicionDA.store_condicion(CondicionType.Max_HR, max_hr)
        condicionId7 = condicionDA.store_condicion(CondicionType.EXANG, exang)
        condicionId8 = condicionDA.store_condicion(CondicionType.Vessels, vessels)
        condicionId9 = condicionDA.store_condicion(CondicionType.Thal, thal)
        condicionId10 = condicionDA.store_condicion(CondicionType.Slope, slope)
        condicionId11 = condicionDA.store_condicion(CondicionType.Oldpeak, oldpeak)
        

        conditions = (condicionId1, condicionId2, condicionId3, condicionId4, condicionId5, condicionId6, condicionId7, condicionId8, condicionId9, condicionId10, condicionId11)
        
        #Conditions needed for current predictive model
        #(condicionId1, condicionId2, condicionId3, condicionId4, condicionId5, condicionId6, condicionId7, condicionId10, condicionId11)
        model_conditions = {
            "cp" : condicionId1,
            "rbp" : condicionId2,
            "chol" : condicionId3,
            "fbs" : condicionId4,
            "rest_ecg" : condicionId5,
            "max_hr" : condicionId6,
            "exang" : condicionId7,
            "slope" : condicionId10,
            "oldpeak" : condicionId11
        }

        # Convert the dictionary to a JSON string
        param1 = json.dumps(model_conditions)
        
        #Create a instance of table VISITA_CONDICION for each CONDICION with the VISITA created
        for cond in conditions:
            condicionDA.store_visita_condicion(cond, visitId)

        #Get the attributes needed from the patient for the prediction
        patient = patientDA.get_patient_by_id(patientId)
        #patient_details = (patient.get_age(), patient.get_sexo(), visitId)
        patient_details = {
            "age" : patient.get_age(),
            "sex" : patient.get_sexo(),
            "visitId" : visitId
        }

        # Convert the dictionary to a JSON string
        param2 = json.dumps(patient_details)

        #Redirect back to record with the new visit
        return redirect(url_for('prediction.predict', conditions=param1, patient_details=param2))
    return redirect(url_for('views.main'))