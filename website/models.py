from flask_login import UserMixin


class Note(db.Model):
    id = dbColumn(db.Integer, primary_key=True)
    data = db.Column(db.String(1000))
    date = dbColumn(db.DateTime(timezone=True), default=func.now())

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db>column(db.String(150))
