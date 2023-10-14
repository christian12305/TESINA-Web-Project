class RecordMedico:
    def __init__(self, id, id_paciente):
        self.__set_id(id)
        self.__set_id_paciente(id_paciente)

    # Getter and Setter for id
    def get_id(self):
        return self.id

    def __set_id(self, id):
        self.id = id

    # Getter and Setter for id_paciente
    def get_id_paciente(self):
        return self.id_paciente

    def __set_id_paciente(self, id_paciente):
        self.id_paciente = id_paciente