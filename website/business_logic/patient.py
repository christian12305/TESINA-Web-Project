from datetime import datetime

class Patient:
    def __init__(self, id, primer_nombre, inicial, apellido_paterno, apellido_materno, fecha_nacimiento, sexo, peso, condicion, correo_electronico, celular):
        self.__set_id(id)
        self.__set_primer_nombre(primer_nombre)
        self.__set_inicial(inicial)
        self.__set_apellido_paterno(apellido_paterno)
        self.__set_apellido_materno(apellido_materno)
        self.__set_fecha_nacimiento(fecha_nacimiento)
        self.__set_sexo(sexo)
        self.__set_peso(peso)
        self.__set_condicion(condicion)
        self.__set_correo_electronico(correo_electronico)
        self.__set_celular(celular)

    # Getter and Setter for id
    def get_id(self):
        return self.id

    def __set_id(self, id):
        self.id = id

    # Getter and Setter for primer_nombre
    def get_primer_nombre(self):
        return self.primer_nombre

    def __set_primer_nombre(self, primer_nombre):
        self.primer_nombre = primer_nombre

    # Getter and Setter for inicial
    def get_inicial(self):
        return self.inicial

    def __set_inicial(self, inicial):
        self.inicial = inicial

    # Getter and Setter for apellido_paterno
    def get_apellido_paterno(self):
        return self.apellido_paterno

    def __set_apellido_paterno(self, apellido_paterno):
        self.apellido_paterno = apellido_paterno

    # Getter and Setter for apellido_materno
    def get_apellido_materno(self):
        return self.apellido_materno

    def __set_apellido_materno(self, apellido_materno):
        self.apellido_materno = apellido_materno

    # Getter and Setter for fecha_nacimiento
    def get_fecha_nacimiento(self):
        return self.fecha_nacimiento

    def __set_fecha_nacimiento(self, fecha_nacimiento):
        self.fecha_nacimiento = fecha_nacimiento

    # Getter and Setter for sexo
    def get_sexo(self):
        if self.sexo == 1:
            return 'Male'
        else:
            return 'Female'

    def __set_sexo(self, sexo):
        self.sexo = sexo

    # Getter and Setter for peso
    def get_peso(self):
        return self.peso

    def __set_peso(self, peso):
        self.peso = peso

    # Getter and Setter for condicion
    def get_condicion(self):
        return self.condicion

    def __set_condicion(self, condicion):
        self.condicion = condicion

    # Getter and Setter for correo_electronico
    def get_correo_electronico(self):
        return self.correo_electronico

    def __set_correo_electronico(self, correo_electronico):
        self.correo_electronico = correo_electronico

    # Getter and Setter for celular
    def get_celular(self):
        return self.celular

    def __set_celular(self, celular):
        self.celular = celular

    #Method to get age
    def get_age(self):
        birthDate = self.get_fecha_nacimiento()
        today = datetime.now()
        age = today.year - birthDate.year
    
        # Check if the birthdate has occurred this year already
        if (today.month, today.day) < (birthDate.month, birthDate.day):
            age -= 1
    
        return age