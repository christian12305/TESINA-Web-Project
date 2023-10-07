from flask import Blueprint, render_template, request, redirect, url_for, session
from . import db
import MySQLdb.cursors

views = Blueprint('views', __name__)

#################
# General views #
#################

#View route for the home screen
@views.route('/', methods=['GET', 'POST'])
def home():
    if 'loggedin' in session:
        return render_template("home.html", session=session)
    return render_template("main.html")

#
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
        visitId = request.args.get('visitId')
        return render_template('result_analysis.html', visitId=visitId)
    return redirect(url_for('auth.login'))

@views.route('/predict')
def predict():
    if 'loggedin' in session:
        pass
    return redirect(url_for('auth.login'))