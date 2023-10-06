from .. import db
from ..models import Visita

class VisitDataAccess:

    def __init__(self):
        self.db_connection = db

    #Extracts from the database the patient by their id
    def get_patient_visits(self, patient_id):

        ##Creating a connection cursor
        cursor = self.db_connection.connection.cursor()
        
        cursor.execute(''' SELECT DISTINCT v.num_visita, v.fecha_visita, e.resultado_evaluacion FROM PACIENTE p JOIN RECORD_MEDICO r on p.id_pk = r.id_paciente_fk JOIN VISITA v on r.id_pk = v.id_pk JOIN RESULTADO e on v.id_resultado_fk = e.id_pk WHERE p.id_pk = %s ORDER BY v.fecha_visita DESC; ''' , (patient_id))
        # Fetch one record and return the result
        visits = cursor.fetchall()

        # Close the cursor
        cursor.close()

        # If patient_data is None, return None
        if not visits:
            return None

        # Create and return a Patient instance with the fetched data
        return visits

    
    #Inserts the patient with the given inputs
    def store_patient(self, firstName, initial, firstLastName, secondLastName, birthDate, gender, weight, condition, email, celullar):
        
        ##Creating a connection cursor
        cursor = self.db_connection.connection.cursor()
        cursor.execute(''' INSERT INTO PACIENTE (primer_nombre, inicial, apellido_paterno, apellido_materno, fecha_nacimiento, sexo, peso, condicion, correo_electronico, celular) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ''',(firstName, initial, firstLastName, secondLastName, birthDate, gender, weight, condition, email, celullar))
        #Saving the Actions performed on the DB
        db.connection.commit()

        #Get the same instance, to then return the id
        patient = self.get_patient_by_email(email)

        # Close the cursor
        cursor.close()

        return patient.get_id()