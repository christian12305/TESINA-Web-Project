from flask import Blueprint, render_template, request, redirect, url_for, session
from .dataAccess.userDA import UserDataAccess

views = Blueprint('views', __name__)

#Init DA classess
userDA = UserDataAccess()

#################
# General views #
#################

#View route for the home screen
@views.route('/', methods=['GET', 'POST'])
def home():
    if 'loggedin' in session:
        return render_template("home.html", session=session)
    return render_template("main.html")

#View route for profile
@views.route('/profile', methods=['POST'])
def profile():
    if 'loggedin' in session:
        user = userDA.get_user_by_id(session['id'])
        return render_template("profile.html", user=user)
    return redirect(url_for('auth.login'))