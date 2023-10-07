from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from .dataAccess.patientDA import PatientDataAccess
from .dataAccess.visitDA import VisitDataAccess
from .dataAccess.condicionDA import CondicionDataAccess
from .dataAccess.recordDA import RecordDataAccess
from .models import CondicionType

patient = Blueprint('patient', __name__)

#Init DA class
patientDA = PatientDataAccess()
visitDA = VisitDataAccess()
condicionDA = CondicionDataAccess()
recordDA = RecordDataAccess()


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
    return redirect(url_for('auth.login'))

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
                #Notify the user
                flash('Patient created!', category='s')
                #Send to the new patient record
                return redirect(url_for('patient.patient_record', patientId=patientId))
        #GET method
        return render_template("create_patient.html")
    return redirect(url_for('auth.login'))


#View route for the patient record endpoint
@patient.route('/patient_record', methods=['GET'])
def patient_record():
    #Verify if user is loggedin
    if 'loggedin' in session:
        patientId = request.args.get('patientId')
        patient = patientDA.get_patient_by_id(patientId)
        visits = visitDA.get_patient_visits(patientId)
        return render_template('patient_record.html', patient=patient, visits=visits)
    return redirect(url_for('auth.login'))
 
#View route for the new visit endpoint
@patient.route('/new_visit', methods=['GET', 'POST'])
def new_visit():
    if 'loggedin' in session:

        if request.method == 'GET':
            recordId = request.args.get('recordId')
            patientId = request.args.get('patientId')
            return render_template('new_visit.html', recordId=recordId, patientId=patientId)

        #Obtain all form inputs

        #Hidden inputs
        patientId = request.form['patientId']
        recordId = request.form['recordId']

        cp = request.form['chest_pain']
        rbp = request.form['resting_bp']
        chol = request.form['cholesterol']
        fbs = request.form['fasting_sugar']
        exang = request.form['exang']
        max_hr = request.form['max_heart_rate']
        vessels = request.form['major_vessels']
        thal = request.form['thal']
        slope = request.form['slope']
        oldpeak = request.form['oldpeak']
        rest_ecg = request.form['rest_ecg']

        #Create a VISITA instance
        visitId = visitDA.new_visit(recordId)

        #Create a instance of table CONDICION and
        # table VISITA_CONDICION for each attribute with the VISITA created

        condicionId1 = condicionDA.store_condicion(CondicionType.ChestPain, cp)
        id_visit_condicionId1 = condicionDA.store_visita_condicion(condicionId1, visitId)

        condicionId2 = condicionDA.store_condicion(CondicionType.RBP, rbp)
        id_visit_condicionId2 = condicionDA.store_visita_condicion(condicionId2, visitId)

        condicionId3 = condicionDA.store_condicion(CondicionType.Chol, chol)
        id_visit_condicionId3 = condicionDA.store_visita_condicion(condicionId3, visitId)

        condicionId4 = condicionDA.store_condicion(CondicionType.FBS, fbs)
        id_visit_condicionId4 = condicionDA.store_visita_condicion(condicionId4, visitId)

        condicionId5 = condicionDA.store_condicion(CondicionType.EXANG, exang)
        id_visit_condicionId5 = condicionDA.store_visita_condicion(condicionId5, visitId)

        condicionId6 = condicionDA.store_condicion(CondicionType.Max_HR, max_hr)
        id_visit_condicionId6 = condicionDA.store_visita_condicion(condicionId6, visitId)

        condicionId7 = condicionDA.store_condicion(CondicionType.Vessels, vessels)
        id_visit_condicionId7 = condicionDA.store_visita_condicion(condicionId7, visitId)

        condicionId8 = condicionDA.store_condicion(CondicionType.Thal, thal)
        id_visit_condicionId8 = condicionDA.store_visita_condicion(condicionId8, visitId)

        condicionId9 = condicionDA.store_condicion(CondicionType.Slope, slope)
        id_visit_condicionId9 = condicionDA.store_visita_condicion(condicionId9, visitId)

        condicionId10 = condicionDA.store_condicion(CondicionType.Oldpeak, oldpeak)
        id_visit_condicionId10 = condicionDA.store_visita_condicion(condicionId10, visitId)

        condicionId11 = condicionDA.store_condicion(CondicionType.RestECG, rest_ecg)
        id_visit_condicionId11 = condicionDA.store_visita_condicion(condicionId11, visitId)

        #Produce a result for the given inputs
        #redirect(url_for(''))

        #Redirect back to record with the new visit
        return redirect(url_for('patient.patient_record', patientId=patientId))
    return redirect(url_for('auth.login'))