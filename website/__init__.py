from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
from flask_mysqldb import MySQL

db = SQLAlchemy()
DB_NAME = "database.db"


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'

    #app.config['MYSQL_HOST'] = 'localhost'
    #app.config['MYSQL_USER'] = 'root'
    #app.config['MYSQL_PASSWORD'] = ''
    #app.config['MYSQL_DB'] = 'flask'

    #db = MySQL(app)
    ##Creating a connection cursor
    #cursor = mysql.connection.cursor()

    #Executing SQL Statements
    #cursor.execute(''' CREATE TABLE table_name(field1, field2...) ''')
    #cursor.execute(''' INSERT INTO table_name VALUES(v1,v2...) ''')
    #cursor.execute(''' DELETE FROM table_name WHERE condition ''')

    #Saving the Actions performed on the DB
    #mysql.connection.commit()
    #Closing the cursor
    #cursor.close()

    db.init_app(app)

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User, Patient
    
    with app.app_context():
        db.create_all()

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app


def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')