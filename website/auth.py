from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from .dataAccess.userDA import UserDataAccess
from datetime import datetime

auth = Blueprint('auth', __name__)

userDA = UserDataAccess()


#################
#   Auth views  #
#################

#Views route for the login endpoint
@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        #Receive values to predict
        email = request.form['email']
        password = request.form['password']

        #Check if user exists
        user = userDA.get_user_by_email(email)
        if user:
            #Password validation
            if check_password_hash(user.get_contraseña(), password):
                flash('Logged in successfully!', category='s')
                #Store session
                session['loggedin'] = True
                session['id'] = user.get_id()
                session['username'] = user.get_correo_electronico()
                # Set session time to current time
                session['_session_time'] = datetime.utcnow()
                return redirect(url_for('views.home'))
            else:
                flash('Incorrect password, try again.', category='e')
        else:
            flash('Email does not exist.', category='e')
    #GET
    return render_template("login.html")


#Views route for the logout endpoint
@auth.route('/logout', methods=['GET'])
def logout():
    if 'loggedin' in session:
        # Remove session data, this will log the user out
        session.pop('loggedin', None)
        session.pop('id', None)
        session.pop('username', None)
        return redirect(url_for('auth.login'))
    return redirect(url_for('auth.login'))

#Views route for the sign up endpoint
@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':

        #Receive user inputs
        email = request.form['email']
        first_name = request.form['firstName']
        initial = request.form['initial']
        last_name = request.form['lastName']
        password1 = request.form['password1']
        password2 = request.form['password2']

        # Check if passwords match
        if password1 != password2:
            return render_template(url_for('auth.sign_up', error="Passwords do not match"))
            #return render_template('signup.html', error="Passwords do not match")

        #Check if user doesnt exists
        user = userDA.get_user_by_email(email)
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

            #Save the user with given inputs
            params = (first_name, initial, last_name, email, password)
            user = userDA.store_user(*params)

            #Log in the user
            session['loggedin'] = True
            session['id'] = user.get_id()
            session['username'] = user.get_correo_electronico()

            flash('Account created!', category='s')
            return redirect(url_for('views.home'))
    #GET
    return render_template("sign_up.html")