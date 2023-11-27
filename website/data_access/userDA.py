from .. import db
from ..business_logic.user import User

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

        # If user is None, return None
        if not user:
            return None
        
        # Initialize and return a User instance with the fetched data
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

        # If user is None, return None
        if not user:
            return None
        
        # Initialize and return a User instance with the fetched data
        return User(*user)
            

    #Inserts the patient with the given inputs
    def store_user(self, first_name, initial, last_name, email, password, rol):

        #Correct row input
        if(rol == "1"):
            rol = 1
        else:
            rol = 2
        ##Creating a connection cursor
        cursor = db.connection.cursor()
        cursor.execute(''' INSERT INTO USUARIO(primer_nombre, inicial, apellido, correo_electronico, contraseña, id_rol_fk) VALUES(%s, %s, %s, %s, %s, %s) ''', (first_name, initial, last_name, email, password, rol,))
        #Saving the Actions performed on the DB
        db.connection.commit()
        # Close the cursor
        cursor.close()

        #Get and return the user stored
        user = self.get_user_by_email(email)
        return user
        
    #Returns all the users in the database
    def getUsers(self):

        #Creating a connection cursor
        cursor = db.connection.cursor()
        cursor.execute(''' SELECT * FROM USUARIO''')
        # Fetch all users
        results = cursor.fetchall()
        #Closing the cursor
        cursor.close()
        # If results is None, return None
        if not results:
            return None
        
        return results
    
    #Updates the user
    def update_user(self, userId, first_name, initial, last_name, email, id_rol, password=False):

        #Creating a connection cursor
        cursor = db.connection.cursor()
        try:
            #If password has been changed
            if password:
                #Update the user
                cursor.execute(''' UPDATE USUARIO set primer_nombre = %s, inicial = %s, apellido = %s, correo_electronico = %s, contraseña = %s, id_rol_fk = %s WHERE id_pk = %s''', (first_name, initial, last_name, email, password, id_rol, userId,))
            else:
                #Update the user
                cursor.execute(''' UPDATE USUARIO set primer_nombre = %s, inicial = %s, apellido = %s, correo_electronico = %s, id_rol_fk = %s WHERE id_pk = %s''', (first_name, initial, last_name, email, id_rol, userId,))
        except Exception as e:
            print(f"Error: {e}")

        finally:
            #Closing the cursor
            cursor.close()