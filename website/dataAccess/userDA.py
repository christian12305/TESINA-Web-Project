from .. import db
from ..business_logic.models import User

class UserDataAccess:

    def __init__(self):
        self.db_connection = db
    
    #Extracts from the database the user by their id
    def get_user_by_id(self, user_id):
        ##Creating a connection cursor
        cursor = db.connection.cursor()
        cursor.execute('''SELECT * FROM USUARIO WHERE id_pk = %s''', (user_id,))
        user = cursor.fetchone()
        #Closing the cursor
        cursor.close()

        if not user:
            return None

        return User(*user)
    
    #Extracts from the database the user by their email
    def get_user_by_email(self, email):
        # Check if account exists using MySQL
        cursor = db.connection.cursor()
        cursor.execute('''SELECT * FROM USUARIO WHERE correo_electronico = %s''', (email,))
        # Fetch one record and return the result
        user = cursor.fetchone()
        #Closing the cursor
        cursor.close()

        if not user:
            return None

        return User(*user)


    #Inserts the patient with the given inputs
    def store_user(self, first_name, initial, last_name, email, password):
        
        ##Creating a connection cursor
        cursor = db.connection.cursor()
        cursor.execute(''' INSERT INTO USUARIO(primer_nombre, inicial, apellido_paterno, correo_electronico, contraseña) VALUES(%s, %s, %s, %s, %s) ''', (first_name, initial, last_name, email, password,))
        #Saving the Actions performed on the DB
        db.connection.commit()

        user = self.get_user_by_email(email)

        # Close the cursor
        cursor.close()

        return user
        

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
    


