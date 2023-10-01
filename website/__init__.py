from flask import Flask
from os import path
from flask_mysqldb import MySQL
from flask_session import Session
#rom flask_jsglue import JSGlue

db = MySQL()

    ##Creating a connection cursor
    #cursor = mysql.connection.cursor()
    #cursor.execute(''' INSERT INTO table_name VALUES(v1,v2...) ''')
    #Saving the Actions performed on the DB
    #mysql.connection.commit()
    #Closing the cursor
    #cursor.close()

    #Executing SQL Statements
    #cursor.execute(''' CREATE TABLE table_name(field1, field2...) ''')
    #cursor.execute(''' INSERT INTO table_name VALUES(v1,v2...) ''')
    #cursor.execute(''' DELETE FROM table_name WHERE condition ''')

    #Saving the Actions performed on the DB
    #mysql.connection.commit()
    #Closing the cursor
    #cursor.close()


def create_app():
    app = Flask(__name__)
    
    app.config['SECRET_KEY'] = 'TESINA-SICI4038'
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = 'Christi@nn23'
    app.config['MYSQL_DB'] = 'tesina'
    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_TYPE"] = "filesystem"
    Session(app)

    db.init_app(app)

    #jsglue = JSGlue()

    #jsglue.init_app(app)

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    #from .models import User, Patient
    #with app.app_context():
    #    db.create_all()

    return app


#def create_database(app):
#    if not path.exists('website/' + DB_NAME):
#        db.create_all(app=app)
#        print('Created Database!')

def getPatients(input):
    ##Creating a connection cursor
    cursor = db.connection.cursor()
    cursor.execute(''' SELECT * FROM PACIENTE WHERE LOWER(primer_nombre) LIKE LOWER(%s) OR LOWER(apellido_paterno) LIKE LOWER(%s) OR LOWER(correo_electronico) LIKE LOWER(%s)''', (input, input, input))

    results = cursor.fetchall()
    #Closing the cursor
    cursor.close()
    return results