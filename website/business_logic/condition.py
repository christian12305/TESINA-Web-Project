class Condicion:
    def __init__(self, id, tipo_condicion, cantidad):
        self.__set_id(id)
        self.__set_tipo_condicion(tipo_condicion)
        self.__set_cantidad(cantidad)

    # Getter and Setter for id
    def get_id(self):
        return self.id

    def __set_id(self, id):
        self.id = id

    # Getter and Setter for tipo_condicion
    def get_tipo_condicion(self):
        return self.tipo_condicion

    def __set_tipo_condicion(self, tipo_condicion):
        self.tipo_condicion = tipo_condicion

    # Getter and Setter for cantidad
    def get_cantidad(self):
        return self.cantidad

    def __set_cantidad(self, cantidad):
        self.cantidad = cantidad