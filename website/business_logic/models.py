#######################################################################################
# This models.py contains all the classes for the entities which we will be using    ##
#
#######################################################################################
from datetime import datetime
from enum import Enum

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
        return self.sexo

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

class User:
    def __init__(self, id, primer_nombre, inicial, apellido_paterno, correo_electronico, contraseña):
        self.__set_id(id)
        self.__set_primer_nombre(primer_nombre)
        self.__set_inicial(inicial)
        self.__set_apellido_paterno(apellido_paterno)
        self.__set_correo_electronico(correo_electronico)
        self.__set_contraseña(contraseña)

    # Getter and Setter for id
    def get_id(self):
        return self.id

    def __set_id(self, id):
        self.id = id

    # Getter and Setter for nombre
    def get_primer_nombre(self):
        return self.primer_nombre

    def __set_primer_nombre(self, primer_nombre):
        self.primer_nombre = primer_nombre

    # Getter and Setter for inicial
    def get_inicial(self):
        return self.inicial

    def __set_inicial(self, inicial):
        self.inicial = inicial

    # Getter and Setter for apellido paterno
    def get_apellido_paterno(self):
        return self.apellido_paterno

    def __set_apellido_paterno(self, apellido_paterno):
        self.apellido_paterno = apellido_paterno

    # Getter and Setter for correo electronico
    def get_correo_electronico(self):
        return self.correo_electronico

    def __set_correo_electronico(self, correo_electronico):
        self.correo_electronico = correo_electronico

    # Getter and Setter for contraseña
    def get_contraseña(self):
        return self.contraseña

    def __set_contraseña(self, contraseña):
        self.contraseña = contraseña



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


class CondicionType(Enum):
    #Angia, chest pain
    ChestPain = 3
    #Resting Blood Pressure
    RBP = 4
    #Cholesterol
    Chol = 5
    #Fasting Blood Sugar
    FBS = 6
    #Resting electrocardiogram results
    RestECG = 7
    #Maximum Heart Rate Achieved
    Max_HR = 8
    #Excercise induced Angia
    EXANG = 9
    #Meassurement of the ST segment depression
    Oldpeak = 10
    #Slope of the peak excercise ST segment
    Slope = 11
    #Number of major vessels
    Vessels = 12
    #Thalassemia
    Thal = 13


class Condicion:
    def __init__(self, id, tipo_condicion, nombre, cantidad, condicion_pk):
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


class VisitaCondicion:
    def __init__(self, id, id_condicion, id_visita):
        self.__set_id(id)
        self.__set_id_condicion(id_condicion)
        self.__set_id_visita(id_visita)


    # Getter and Setter for id
    def get_id(self):
        return self.id

    def __set_id(self, id):
        self.id = id

    # Getter and Setter for id_condicion
    def get_id_condicion(self):
        return self.id_condicion

    def __set_id_condicion(self, id_condicion):
        self.id_condicion = id_condicion
    
    # Getter and Setter for id_visita_fk
    def get_id_visita(self):
        return self.id_visita

    def __set_id_visita(self, id_visita):
        self.id_visita = id_visita



class Visita:
    def __init__(self, id, num_visita, fecha, id_visita_condicion, id_resultado, id_record_medico):
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
