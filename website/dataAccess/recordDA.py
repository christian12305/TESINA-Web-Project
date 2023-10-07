from .. import db

class RecordDataAccess:

    def __init__(self):
        self.db_connection = db

    #Extracts from the database the patient by their id
    def get_patient_by_record(self, recordId):

        ##Creating a connection cursor
        cursor = self.db_connection.connection.cursor()
        cursor.execute('''SELECT * FROM RECORD_MEDICO WHERE id_pk = %s''' , (recordId,))
        # Fetch one record and return the result
        record = cursor.fetchone()

        # Close the cursor
        cursor.close()

        # If patient_data is None, return None
        if not record:
            return None

        # Create and return a Patient instance with the fetched data
        return record[1]