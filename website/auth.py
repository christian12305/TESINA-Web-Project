from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from .data_access.userDA import UserDataAccess
from datetime import datetime
import re

auth = Blueprint('auth', __name__)

userDA = UserDataAccess()


#################
#   Auth views  #
#################

#Auxiliary function for user validation
#This method performs validations on users
def isValid(email, first_name, initial, last_name, password1, password2, edit=False):
    #Init
    err = ''
    valid = False

    # Check if user doesnt exists
    user = userDA.get_user_by_email(email)

    if user and not edit:
        err = 'Email already exists.'
        return (valid, err)
  
    #Validations
    #Email must be {words and . and -} @ {words . and -} . {words}>2
    email_regex = re.compile(r'^[\w\.-]+@[a-zA-Z\d\.-]+\.[a-zA-Z]{2,}$')
    # Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, and one digit.
    password_regex = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$')
    #First and Last name regex
    name_regex = re.compile(r'^[a-zA-Z]+$')
    #Initial regex
    initial_regex = re.compile(r'^[A-Z][/.]$')

    if not bool(email_regex.match(email)):
        err = 'Email is not valid.'
        return (valid, err)
    elif len(first_name) < 2:
        err = 'First name must be greater than 1 character.'
        return (valid, err)
    elif not bool(name_regex.match(first_name)):
        err='First name is not valid.'
        return (valid, err)
    elif initial and not bool(initial_regex.match(initial)):
        err = 'Initial must contain only 1 character and a dot (.)'
        return (valid, err)
    elif len(last_name) < 2:
        err = 'Last name must be greater than 1 character.'
        return (valid, err)
    elif not bool(name_regex.match(last_name)):
        err='Last name is not valid.'
        return (valid, err)
    elif password1 != password2:
        err = 'Passwords don\'t match.'
        return (valid, err)
    elif password1 and not bool(password_regex.match(password1)):
        err = 'Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, and one digit.'
        return (valid, err)
    else:
        valid = True
        return (valid, err)

#Auxiliary function to store the session of user when logging in
def session_login(user):
    #Log in the user and store session
    session['loggedin'] = True
    session['id'] = user.get_id()
    session['username'] = user.get_correo_electronico()
    if user.get_rol() == 'admin':
        session['role'] = True
    else:
        session['role'] = False
    # Set session time to current time
    session['_session_time'] = datetime.utcnow()

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
                session_login(user)
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
        session.pop('role', None)
        return redirect(url_for('views.home'))
    return redirect(url_for('views.main'))

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
        rol = request.form['accountType']

        # Check if passwords match
        if password1 != password2:
            return render_template('sign_up.html', error="Passwords do not match")

        #Validations
        user = userDA.get_user_by_email(email)

        (valid, err) = isValid(email, first_name, initial, last_name, password1, password2)
        if not valid:
          flash(err, category='e')
        else:
        
            #Generates a hashed password
            password=generate_password_hash(password1)

            #Save the user with given inputs
            params = (first_name, initial, last_name, email, password, rol)
            user = userDA.store_user(*params)

            session_login(user)

            flash('Account created!', category='s')
            return redirect(url_for('views.home'))
    #GET
    return render_template("sign_up.html")

@auth.route('/edit_user', methods=['GET', 'POST'])
def edit_user():
    #Verify if user is loggedin and is admin
    if 'loggedin' in session and 'role' in session:
        #POST
        if request.method == 'POST':
            
            #Receive user inputs
            email = request.form['email']
            first_name = request.form['firstName']
            initial = request.form['initial']
            last_name = request.form['lastName']
            password1 = request.form['password1']
            password2 = request.form['password2']
            rol = request.form['accountType']
            userId = request.form['user_id']

            #If password change
            if password1 != password2:
                error="Passwords do not match."
                flash(error, category='e')
                
            (valid, err) = isValid(email, first_name, initial, last_name, password1, password2, True)
            if not valid:
                flash(err, category='e')
            else:
                #If password has been changed
                if password1:
                    #Generates a hashed password
                    password=generate_password_hash(password1)
                    userDA.update_user(userId, first_name, initial, last_name, email, rol, password)
                else:
                    userDA.update_user(userId, first_name, initial, last_name, email, rol)
                flash('Account saved!', category='s')        
        #GET
        user_id = request.args.get('user_id')
        user = userDA.get_user_by_id(user_id)

        if user and user.get_id() == userDA.get_user_by_id(session['id']).get_id() or session['role']:
            return render_template('edit_user.html', user=user, admin=session['role'])
        else:
            return redirect(url_for('views.home'))
    return redirect(url_for('views.home'))