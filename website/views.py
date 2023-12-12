from flask import Blueprint, render_template, redirect, url_for, session
from .data_access.userDA import UserDataAccess

views = Blueprint('views', __name__)

#Init DA classess
userDA = UserDataAccess()

#################
# General views #
#################

#View route for the home screen
@views.route('/home', methods=['GET', 'POST'])
def home():
    if 'loggedin' in session:
        return render_template("home.html")
    return redirect(url_for('views.main'))

#View route for the main screen
@views.route('/', methods=['GET'])
def main():
    return render_template("main.html")

#View route for profile
@views.route('/profile', methods=['GET'])
def profile():
    if 'loggedin' in session:
        user = userDA.get_user_by_id(session['id'])
        return render_template("profile.html", user=user)
    return redirect(url_for('views.home'))

@views.route('/config', methods=['GET', 'POST'])
def config():
    #Verify if user is loggedin and is admin
    if 'loggedin' in session and session['role']:
        users = userDA.getUsers()
        return render_template('config.html', users=users)
    return redirect(url_for('views.home'))