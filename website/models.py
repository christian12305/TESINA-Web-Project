from . import db
#from flask_login import UserMixin
#rom sqlalchemy.sql import func

'''
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lastName = db.Column(db.String(30))
    initial = db.Column(db.String(2))
    firstName = db.Column(db.String(30))
    gender = db.Column(db.String(10))
    weight = db.Column(db.String(4))
    condition = db.Column(db.String(20))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    #notes = db.relationship('Note')
'''