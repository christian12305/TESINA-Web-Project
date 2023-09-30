from flask import Blueprint, render_template, request, flash, jsonify, redirect, url_for
from flask_login import login_required, current_user
from .models import Patient
from . import db
import json

views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST': 
        #note = request.form.get('note')#Gets the note from the HTML 

        #if len(note) < 1:
        #   flash('Note is too short!', category='error') 
        #else:
            #new_note = Note(data=note, user_id=current_user.id)  #providing the schema for the note 
            #db.session.add(new_note) #adding the note to the database 
            #db.session.commit()
            #flash('Note added!', category='success')
        pass

    return render_template("home.html", user=current_user)


@views.route('/search', methods=['GET', 'POST'])
@login_required
def search_patient():
    if request.method == 'POST':
        #Received a search request
        pass
    return render_template("search_patient.html", user=current_user)


@views.route('/create', methods=['GET', 'POST'])
@login_required
def create_patient():
    if request.method == 'POST':
        #Patient has been submitted for creation
        firstName = request.form.get('firstName')
        intial = request.form.get('initial')
        lastName = request.form.get('lastName')
        gender = request.form.get('gender')
        weight = request.form.get('weight')
        conditon = request.form.get('condition')
        #patient = Patient.query.filter_by(firstName=firstName, intial=intial, lastName=lastName).first()
        #if patient:
            #Conditions to be met
            #elif
            #elif
            #else
            #if all conditions are met, create patient
        #    pass
        return redirect(url_for('views.patient_record'))
        #return render_template("patient_record.html", user=current_user)

        
    return render_template("create_patient.html", user=current_user)


@views.route('/patient-record', methods=['GET', 'POST'])
@login_required
def patient_record():
    if request.method == 'GET':
        return render_template("patient_record.html", user=current_user)
        #return render_template("patient_record.html", patient=patient)









'''
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