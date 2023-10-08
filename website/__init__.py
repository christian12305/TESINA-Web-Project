from flask import Flask
from flask_mysqldb import MySQL
from flask_session import Session
from datetime import timedelta
from .business_logic.prediction.data.data_prep import ModelData
from .business_logic.prediction.prediction_model import PredictionModel

DATA_PATH = "website/business_logic/prediction/data/model_data/unified_data.csv"
modelData = ModelData(DATA_PATH)

db = MySQL()

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
    app.config['SESSION_COOKIE_DURATION'] = timedelta(minutes=5)

    #Initialize the session
    Session(app)

    #Initialize app with the configurations
    db.init_app(app)

    #Registering the views
    from .views import views
    from .auth import auth
    from .patient import patient
    from .prediction import prediction

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    app.register_blueprint(patient, url_prefix='/')
    app.register_blueprint(prediction, url_prefix='/')

    return app

#This method creates the predictive model,
# and trains it for the system to use and predict.
def create_model():
    #Data initialization
    (categorical_features, continuous_features, target) = modelData.get_features()

    #Model initialization
    pred_model = PredictionModel(continuous_features, categorical_features, target)
    return pred_model

#Method to train a model that has just initialized
def train_model():
    model = create_model()
    (train, _, validation) = modelData.get_data()
    model.fit_model(train, validation)
    return model

model_pred = train_model()