from flask import Flask
from flask_mysqldb import MySQL
from flask_session import Session
from datetime import timedelta
from .business_logic.prediction.data.data_prep import ModelData
from .business_logic.prediction.prediction_model import PredictionModel

#Training dataset
DATA_PATH = "website/business_logic/prediction/data/model_data/unified_data.csv"
#Initializing the training data class
modelData = ModelData(DATA_PATH)

db = MySQL()

def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'TESINA-SICI4038'
    #DB configurations
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'tesina'
    app.config['MYSQL_PASSWORD'] = 'adminP@$$'
    app.config['MYSQL_DB'] = 'tesina'
    #Session configurations
    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_TYPE"] = "filesystem"
    app.config['SESSION_COOKIE_DURATION'] = timedelta(minutes=3)

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
            
#Method to create the predictive model,
# and trains it for the system to use and predict
def create_model():
    #Data features/attributes
    (categorical_features, continuous_features, target) = modelData.get_features()

    #Model initialization
    pred_model = PredictionModel(continuous_features, categorical_features, target)
    return pred_model

#Method to train a model that has just initialized
def train_model():
    #Call the method to create the model
    model = create_model()
    #Dataset (train, test, validation)
    (train, _, validation) = modelData.get_data()
    #Train model
    model.fit_model(train, validation)
    return model

#Model initialized and trained to export
model_pred = train_model()