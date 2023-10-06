from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from . import db
import MySQLdb.cursors

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    if 'loggedin' in session:
        return render_template("home.html", session=session)
    return render_template("main.html")

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

@views.route('/predictive_analysis', methods=['GET', 'POST'])
def predictive_analysis():
    if 'loggedin' in session:
        if request.method == 'POST':
            pass
        patientId = request.args.get('patientId')
        return render_template('result_analysis.html', patientId=patientId)
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