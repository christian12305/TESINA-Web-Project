from .. import db
from ..models import Patient

class PatientDataAccess:

    def __init__(self):
        self.db_connection = db

    #Extracts from the database the patient by their id
    def get_patient_by_id(self, patient_id):

        ##Creating a connection cursor
        cursor = self.db_connection.connection.cursor()
        cursor.execute('''SELECT * FROM PACIENTE WHERE id_pk = %s''' , (patient_id))
        # Fetch one record and return the result
        patient = cursor.fetchone()

        # Close the cursor
        cursor.close()

        # If patient_data is None, return None
        if not patient:
            return None

        # Create and return a Patient instance with the fetched data
        return Patient(*patient)
    

    #Extracts from the database the patient by their id
    def get_patient_by_email(self, email):

        ##Creating a connection cursor
        cursor = self.db_connection.connection.cursor()
        cursor.execute('''SELECT * FROM PACIENTE WHERE correo_electronico = %s''' , (email,))
        # Fetch one record and return the result
        patient = cursor.fetchone()
        # Close the cursor
        cursor.close()

        # If patient_data is None, return None
        if not patient:
            return None
        
        # Create and return a Patient instance with the fetched data
        return Patient(*patient)
    
    #Inserts the patient with the given inputs
    def store_patient(self, firstName, initial, firstLastName, secondLastName, birthDate, gender, weight, condition, email, celullar):
        
        ##Creating a connection cursor
        cursor = self.db_connection.connection.cursor()
        cursor.execute(''' INSERT INTO PACIENTE (primer_nombre, inicial, apellido_paterno, apellido_materno, fecha_nacimiento, sexo, peso, condicion, correo_electronico, celular) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ''',(firstName, initial, firstLastName, secondLastName, birthDate, gender, weight, condition, email, celullar))
        #Saving the Actions performed on the DB
        db.connection.commit()

        #Get the same instance, to then return the id
        patient = self.get_patient_by_email(email)

        patientId = patient.get_id()

        #Also build a record for the patient
        cursor.execute(''' INSERT INTO RECORD_MEDICO (id_paciente_fk) VALUES(%s) ''',(patientId,))

        # Close the cursor
        cursor.close()

        return patientId
        

    def getPatients(self, input):
        ##Creating a connection cursor
        cursor = db.connection.cursor()
        cursor.execute(''' SELECT * FROM PACIENTE WHERE LOWER(primer_nombre) LIKE LOWER(%s) OR LOWER(apellido_paterno) LIKE LOWER(%s)''', (input, input,))

        results = cursor.fetchall()

        if not results:
            return None

        #Closing the cursor
        cursor.close()
        return results