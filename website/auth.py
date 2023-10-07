from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from . import db 
import MySQLdb.cursors

auth = Blueprint('auth', __name__)

#Views route for the login endpoint
@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        #user = User.query.filter_by(email=email).first()
        # Check if account exists using MySQL
        cursor = db.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('''SELECT * FROM USUARIO WHERE correo_electronico = %s''', (email,))
        # Fetch one record and return the result
        user = cursor.fetchone()

        if user:
            if check_password_hash(user['contraseña'], password):
                flash('Logged in successfully!', category='s')

                session['loggedin'] = True
                session['id'] = user['id_pk']
                session['username'] = user['correo_electronico']
                
                return redirect(url_for('views.home'))
            else:
                flash('Incorrect password, try again.', category='e')
        else:
            flash('Email does not exist.', category='e')
    #GET request
    return render_template("login.html", session=session)


#Views route for the logout endpoint
@auth.route('/logout')
def logout():
    if 'loggedin' in session:
        # Remove session data, this will log the user out
        session.pop('loggedin', None)
        session.pop('id', None)
        session.pop('username', None)
        return redirect(url_for('views.home'))
    return redirect(url_for('views.home'))

#Views route for the sign up endpoint
@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':

        #Receive request inputs
        email = request.form['email']
        first_name = request.form['firstName']
        initial = request.form['initial']
        last_name = request.form['lastName']
        password1 = request.form['password1']
        password2 = request.form['password2']

        # Check if account exists using MySQL
        cursor = db.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('''SELECT * FROM USUARIO WHERE correo_electronico = %s''', (email,))
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


            #Generates a hashed password
            password=generate_password_hash(password1, method='sha1')

            ##Creating a connection cursor
            cursor = db.connection.cursor()
            cursor.execute(''' INSERT INTO USUARIO(primer_nombre, inicial, apellido_paterno, correo_electronico, contraseña) VALUES(%s, %s, %s, %s, %s) ''', (first_name, initial, last_name, email, password,))
            #Saving the Actions performed on the DB
            db.connection.commit()

            cursor.execute('''SELECT * FROM USUARIO WHERE correo_electronico = %s''', (email,))
            # Fetch one record and return the result
            user = cursor.fetchone()

            #Closing the cursor
            cursor.close()

            #Log in the user
            session['loggedin'] = True
            session['id'] = user[0]
            session['username'] = user[4]

            flash('Account created!', category='s')
            
            return redirect(url_for('views.home'))

    return render_template("sign_up.html")