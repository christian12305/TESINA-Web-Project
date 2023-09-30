from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_mysqldb import MySQL
import MySQLdb.cursors


#db = SQLAlchemy()
#DB_NAME = "database.db"
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
    #app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'

    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = ''
    app.config['MYSQL_DB'] = 'tesina'

    db.init_app(app)

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User, Patient
    
    #with app.app_context():
    #    db.create_all()

    #@login_manager.user_loader
    #def load_user(id):
    #    return User.query.get(int(id))
    def load_user(id):
        cursor = db.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id_pk = %s', (id))
        # Fetch one record and return the result
        user = cursor.fetchone()
        return user

    return app


def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')