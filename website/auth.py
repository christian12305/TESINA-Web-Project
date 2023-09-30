from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
#from . import db   ##means from __init__.py import db

import MySQLdb.cursors, re


auth = Blueprint('auth', __name__)


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        #user = User.query.filter_by(email=email).first()
        # Check if account exists using MySQL
        cursor = db.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE correo_electronico = %s', (email))
        # Fetch one record and return the result
        user = cursor.fetchone()

        if user:
            if check_password_hash(user['password'], password):
                flash('Logged in successfully!', category='s')

                session['loggedin'] = True
                session['id'] = user['id_pk']
                session['username'] = user['correo_electronico']
                
                return redirect(url_for('views.home'))
            else:
                flash('Incorrect password, try again.', category='e')
        else:
            flash('Email does not exist.', category='e')

    return render_template("login.html")


@auth.route('/logout')
def logout():
    # Remove session data, this will log the user out
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('auth.login'))


@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        initial = request.form.get('initial')
        last_name = request.form.get('lastName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        # Check if account exists using MySQL
        cursor = db.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE correo_electronico = %s', (email))
        # Fetch one record and return the result
        user = cursor.fetchone()

        if user:
            flash('Email already exists.', category='e')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='e')
        elif len(first_name) < 2:
            flash('First name must be greater than 1 character.', category='e')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='e')
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.', category='e')
        else:


            password=generate_password_hash(password1, method='sha256')
            ##Creating a connection cursor
            cursor = db.connection.cursor()
            cursor.execute(''' INSERT INTO USUARIO VALUES(first_name, initial, last_name, email, password) ''')
            #Saving the Actions performed on the DB
            db.connection.commit()

            cursor.execute('SELECT * FROM accounts WHERE correo_electronico = %s', (email))
            # Fetch one record and return the result
            user = cursor.fetchone()

            #Closing the cursor
            cursor.close()

            session['loggedin'] = True
            session['id'] = user['id_pk']
            session['username'] = user['correo_electronico']

            flash('Account created!', category='success')
            return redirect(url_for('views.home'))

    return render_template("sign_up.html")