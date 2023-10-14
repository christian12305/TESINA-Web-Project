class Resultado:
    def __init__(self, id, resultado_evaluacion, resultado_pk):
        self.__set_id(id)
        self.__set_resultado_evaluacion(resultado_evaluacion)
        self.__set_id_visita(resultado_pk)

    # Getter and Setter for id
    def get_id(self):
        return self.id

    def __set_id(self, id):
        self.id = id

    # Getter and Setter for resultado_evaluacion
    def get_resultado_evaluacion(self):
        return self.resultado_evaluacion

    def __set_resultado_evaluacion(self, resultado_evaluacion):
        self.resultado_evaluacion = resultado_evaluacion

    # Getter and Setter for id_visita_fk
    def get_id_visita(self):
        return self.id_visita

    def __set_id_visita(self, id_visita):
        self.id_visita = id_visita