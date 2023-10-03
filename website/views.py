from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from . import db, getPatients
import MySQLdb.cursors

views = Blueprint('views', __name__)



@views.route('/', methods=['GET', 'POST'])
def home():
    if 'loggedin' in session:
        return render_template("home.html", session=session)
    return render_template("main.html")


@views.route('/search', methods=['GET', 'POST'])
def search_patient():
    if 'loggedin' in session:
        if request.method == 'POST':
            #Received a search request
            data = request.form.get('search')
            #To find records starting with the data entered by the user
            data = data + '%'
            patients = getPatients(data)
            return render_template("search_patient.html", patients=patients)
        return render_template("search_patient.html")
    return redirect(url_for('auth.login'))


@views.route('/create', methods=['GET', 'POST'])
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


            ##Creating a connection cursor
            cursor = db.connection.cursor()
            cursor.execute('''SELECT * FROM PACIENTE WHERE primer_nombre = %s AND correo_electronico = %s AND fecha_nacimiento = %s''', (firstName, email, birthDate))
            # Fetch one record and return the result
            patient = cursor.fetchone()

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

                cursor.execute(''' INSERT INTO PACIENTE (primer_nombre, inicial, apellido_paterno, apellido_materno, fecha_nacimiento, sexo, peso, condicion, correo_electronico, celular) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ''',(firstName, initial, firstLastName, secondLastName, birthDate, gender, weight, condition, email, celullar))
            
                #Saving the Actions performed on the DB
                db.connection.commit()

                #Closing the cursor
                cursor.close()

                flash('Patient created!', category='s')
                return redirect(url_for('views.patient_record'))

        return render_template("create_patient.html")
    return redirect(url_for('auth.login'))

@views.route('/profile', methods=['POST'])
def profile():
    if 'loggedin' in session:
        cursor = db.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('''SELECT * FROM USUARIO WHERE id_pk = %s''', (session['id'],))
        user = cursor.fetchone()
        #Closing the cursor
        cursor.close()
        return render_template("profile.html", user=user)
    return redirect(url_for('auth.login'))

@views.route('/patient_record/<int:patientId>', methods=['GET'])
def patient_record(patientId):
    if 'loggedin' in session:
        return render_template('patient_record.html', patientId=patientId, session=session)
    return redirect(url_for('auth.login'))
 


'''
@views.route('/patient_record', methods=['POST'])
def patient_record():
    patient = json.loads(request.data)
    patientId = patient['patientId']
    print("POST PATIENT ID: " + str(patientId))
    return render_template(('patient_record.html'), patientId=patientId)

    
@views.route('/delete-note', methods=['POST'])
def delete_note():  
    note = json.loads(request.data) # this function expects a JSON from the INDEX.js file 
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})
'''