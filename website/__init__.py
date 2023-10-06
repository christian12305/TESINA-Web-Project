from flask import Flask
from os import path
from flask_mysqldb import MySQL
from flask_session import Session


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
    
    #App configurations
    app.config['SECRET_KEY'] = 'TESINA-SICI4038'
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = 'Christi@nn23'
    app.config['MYSQL_DB'] = 'tesina'
    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_TYPE"] = "filesystem"

    #
    Session(app)

    #Initialize app with the configurations
    db.init_app(app)


    #Registering the views
    from .views import views
    from .auth import auth
    from .patient import patient

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    app.register_blueprint(patient, url_prefix='/')


    return app