from .. import db
from ..business_logic.models import Resultado

class ResultadoDataAccess:

    def __init__(self):
        self.db_connection = db
    
    #Extracts from the database the resultado by their id
    def get_resultado_by_id(self, resultado_id):
        ##Creating a connection cursor
        cursor = db.connection.cursor()
        cursor.execute('''SELECT * FROM RESULTADO WHERE id_pk = %s''', (resultado_id,))
        resultado = cursor.fetchone()
        #Closing the cursor
        cursor.close()

        # Initialize and return a Resultado instance with the fetched data
        return Resultado(*resultado)


    #Inserts the resultado with the given inputs
    def store_resultado(self, resultado_evaluacion, id_visita):
        ##Creating a connection cursor
        cursor = self.db_connection.connection.cursor()
        cursor.execute(''' INSERT INTO RESULTADO (resultado_evaluacion, id_visita_fk) VALUES(%s,%s) ''',(resultado_evaluacion, id_visita))
        #Saving the Actions performed on the DB
        db.connection.commit()
        # Close the cursor
        cursor.close()
        
    


