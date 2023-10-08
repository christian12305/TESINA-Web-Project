from .. import db
from ..business_logic.models import Condicion

class CondicionDataAccess:

    def __init__(self):
        self.db_connection = db

    #Extracts from the database the patient by their id
    def get_condicion_by_id(self, condicion_id):

        ##Creating a connection cursor
        cursor = self.db_connection.connection.cursor()
        cursor.execute('''SELECT * FROM CONDICION WHERE id_pk = %s''' , (condicion_id,))
        # Fetch one record and return the result
        patient = cursor.fetchone()

        # Close the cursor
        cursor.close()

        # If patient_data is None, return None
        if not patient:
            return None

        # Create and return a Patient instance with the fetched data
        return Condicion(*patient)
    
    #Inserts the patient with the given inputs
    def store_condicion(self, conditionType, quantity):

        ##Creating a connection cursor
        cursor = self.db_connection.connection.cursor()
        cursor.execute(''' INSERT INTO CONDICION (tipo_condicion, cantidad) VALUES(%s,%s) ''',(conditionType.value, quantity,))
        #Saving the actions performed on the DB
        db.connection.commit()

        #Get the new CONDICION id
        cursor.execute(''' SELECT LAST_INSERT_ID() ''')
        id_condicion = cursor.fetchone()[0]

        # Close the cursor
        cursor.close()

        return id_condicion

    #Inserts a VISITA_CONDICION instance with the given inputs
    def store_visita_condicion(self, id_condicion, visitaId):

        ##Creating a connection cursor
        cursor = self.db_connection.connection.cursor()

        #Create a new VISITA_CONDICION instance for this CONDICION
        cursor.execute(''' INSERT INTO VISITA_CONDICION (id_condicion_fk, id_visita_fk) VALUES(%s, %s) ''',(id_condicion, visitaId))
        #Saving the actions performed on the DB
        db.connection.commit()

        #Get the new VISITA_CONDICION id
        cursor.execute(''' SELECT LAST_INSERT_ID() ''')
        id_visita_condicion = cursor.fetchone()[0]

        # Close the cursor
        cursor.close()

        return id_visita_condicion



