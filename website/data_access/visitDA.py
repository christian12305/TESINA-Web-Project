from .. import db
from ..business_logic.visit import Visita

class VisitDataAccess:

    def __init__(self):
        self.db_connection = db

    #Extracts from the database the patient by their id the list of visits
    def get_patient_visits(self, patient_id):
        ##Creating a connection cursor
        cursor = self.db_connection.connection.cursor()
        cursor.execute('''SELECT DISTINCT v.fecha_visita, e.resultado_evaluacion, v.id_pk, r.id_pk FROM PACIENTE p JOIN RECORD_MEDICO r on p.id_pk = r.id_paciente_fk JOIN VISITA v on r.id_pk = v.record_medico_fk JOIN VISITA_CONDICION vc on v.id_pk = vc.id_visita_fk JOIN RESULTADO e on v.id_pk = e.id_visita_fk WHERE p.id_pk = %s ORDER BY v.fecha_visita DESC;''' , (patient_id,))
        # Fetch one record and return the result
        visits = cursor.fetchall()
        # Close the cursor
        cursor.close()

        # If visits is None, return None
        if not visits:
            return None

        return visits

    
    #Inserts the patient with the given inputs
    def new_visit(self, record_medicoId):
        ##Creating a connection cursor
        cursor = self.db_connection.connection.cursor()
        cursor.execute(''' INSERT INTO VISITA (fecha_visita, record_medico_fk) VALUES(CURRENT_TIMESTAMP(), %s) ''',(record_medicoId,))
        #Saving the Actions performed on the DB
        db.connection.commit()

        #Get the new VISITA id
        cursor.execute(''' SELECT LAST_INSERT_ID() ''')
        id_visita = cursor.fetchone()[0]

        #Get the record and create an instance
        cursor.execute('''SELECT * FROM VISITA WHERE id_pk = %s''' , (id_visita,))
        # Fetch visit and return the result
        visita = cursor.fetchone()
        # Close the cursor
        cursor.close()

        # If visita is None, return None
        if not visita:
            return None

        # Initialize and return a Visita instance with the fetched data
        return Visita(*visita)
    
    #Method to return patient id for the visit
    def get_patient_id_by_visit(self, visitId):
        ##Creating a connection cursor
        cursor = self.db_connection.connection.cursor()
        #Get the record and create an instance
        cursor.execute('''SELECT r.id_paciente_fk FROM VISITA as v JOIN RECORD_MEDICO as r on v.record_medico_fk = r.id_pk WHERE v.id_pk = %s''' , (visitId,))
        # Fetch visit and return the result
        patientId = cursor.fetchone()
        # Close the cursor
        cursor.close()

        # If patientId is None, return None
        if not patientId:
            return None

        return patientId