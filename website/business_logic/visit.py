class Visita:
    def __init__(self, id, fecha, id_record_medico):
        self.__set_id(id)
        self.__set_fecha(fecha)
        self.__set_id_record_medico(id_record_medico)

    # Getter and Setter for id
    def get_id(self):
        return self.id

    def __set_id(self, id):
        self.id = id

    # Getter and Setter for fecha
    def get_fecha(self):
        return self.fecha

    def __set_fecha(self, fecha):
        self.fecha = fecha

    # Getter and Setter for id_record_medico
    def get_id_record_medico(self):
        return self.id_record_medico

    def __set_id_record_medico(self, id_record_medico):
        self.id_record_medico = id_record_medico