from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from . import db
from .dataAccess.patientDA import PatientDataAccess

patient = Blueprint('patient', __name__)

patientDA = PatientDataAccess()

@patient.route('/search', methods=['GET', 'POST'])
def search_patient():
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

@patient.route('/create', methods=['GET', 'POST'])
def create_patient():
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

            patient = patientDA.get_patient_by_email(email)

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

                params = (firstName, initial, firstLastName, secondLastName, birthDate, gender, weight, condition, email, celullar)
                patientId = patientDA.store_patient(*params)

                flash('Patient created!', category='s')
                return redirect(url_for('patient.patient_record', patientId=patientId))

        return render_template("create_patient.html")
    return redirect(url_for('auth.login'))


@patient.route('/patient_record', methods=['GET'])
def patient_record():
    if 'loggedin' in session:
        patientId = request.args.get('patientId')
        return render_template('patient_record.html', patientId=patientId, session=session)
    return redirect(url_for('auth.login'))
 

@patient.route('/new_visit', methods=['GET', 'POST'])
def new_visit():
    if 'loggedin' in session:
        if request.method == 'GET':
            return render_template('new_visit.html')
        patientId = request.args.get('patientId')
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
        #CREATE THE NEW VISIT HERE WITH THE INPUT
        #ADD IT TO THE PATIENTS RECORD
        #REDIRECT BACK TO THE PATIENT RECORD
        return redirect(url_for('views.patient_record', patientId=patientId))
    return redirect(url_for('auth.login'))