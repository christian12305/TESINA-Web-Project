#######################################################################################
# This models.py contains all the classes for the entities which we will be using    ##
#
#######################################################################################

class Patient:
    def __init__(self, id, primer_nombre, inicial, apellido_paterno, apellido_materno, fecha_nacimiento, sexo, peso, condicion, correo_electronico, celular):
        self.id = id
        self.primer_nombre = primer_nombre
        self.inicial = inicial
        self.apellido_paterno = apellido_paterno
        self.apellido_materno = apellido_materno
        self.fecha_nacimiento = fecha_nacimiento
        self.sexo = sexo
        self.peso = peso
        self.condicion = condicion
        self.correo_electronico = correo_electronico
        self.celular = celular

    # Getter and Setter for id
    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    # Getter and Setter for primer_nombre
    def get_primer_nombre(self):
        return self.primer_nombre

    def set_primer_nombre(self, primer_nombre):
        self.primer_nombre = primer_nombre

    # Getter and Setter for inicial
    def get_inicial(self):
        return self.inicial

    def set_inicial(self, inicial):
        self.inicial = inicial

    # Getter and Setter for apellido_paterno
    def get_apellido_paterno(self):
        return self.apellido_paterno

    def set_apellido_paterno(self, apellido_paterno):
        self.apellido_paterno = apellido_paterno

    # Getter and Setter for apellido_materno
    def get_apellido_materno(self):
        return self.apellido_materno

    def set_apellido_materno(self, apellido_materno):
        self.apellido_materno = apellido_materno

    # Getter and Setter for fecha_nacimiento
    def get_fecha_nacimiento(self):
        return self.fecha_nacimiento

    def set_fecha_nacimiento(self, fecha_nacimiento):
        self.fecha_nacimiento = fecha_nacimiento

    # Getter and Setter for sexo
    def get_sexo(self):
        return self.sexo

    def set_sexo(self, sexo):
        self.sexo = sexo

    # Getter and Setter for peso
    def get_peso(self):
        return self.peso

    def set_peso(self, peso):
        self.peso = peso

    # Getter and Setter for condicion
    def get_condicion(self):
        return self.condicion

    def set_condicion(self, condicion):
        self.condicion = condicion

    # Getter and Setter for correo_electronico
    def get_correo_electronico(self):
        return self.correo_electronico

    def set_correo_electronico(self, correo_electronico):
        self.correo_electronico = correo_electronico

    # Getter and Setter for celular
    def get_celular(self):
        return self.celular

    def set_celular(self, celular):
        self.celular = celular



class User:
    def __init__(self, id, primer_nombre, inicial, apellido_paterno, correo_electronico, contraseña):
        self.id = id
        self.primer_nombre = primer_nombre
        self.inicial = inicial
        self.apellido_paterno = apellido_paterno
        self.correo_electronico = correo_electronico
        self.contraseña = contraseña

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    def get_primer_nombre(self):
        return self.primer_nombre

    def set_primer_nombre(self, primer_nombre):
        self.primer_nombre = primer_nombre

    def get_inicial(self):
        return self.inicial

    def set_inicial(self, inicial):
        self.inicial = inicial

    def get_apellido_paterno(self):
        return self.apellido_paterno

    def set_apellido_paterno(self, apellido_paterno):
        self.apellido_paterno = apellido_paterno

    def get_correo_electronico(self):
        return self.correo_electronico

    def set_correo_electronico(self, correo_electronico):
        self.correo_electronico = correo_electronico

    def get_contraseña(self):
        return self.contraseña

    def set_contraseña(self, contraseña):
        self.contraseña = contraseña



class Resultado:
    def __init__(self, id, resultado_evaluacion, resultado_pk):
        self.id = id
        self.resultado_evaluacion = resultado_evaluacion
        self.resultado_pk = resultado_pk

    # Getter and Setter for id
    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    # Getter and Setter for resultado_evaluacion
    def get_resultado_evaluacion(self):
        return self.resultado_evaluacion

    def set_resultado_evaluacion(self, resultado_evaluacion):
        self.resultado_evaluacion = resultado_evaluacion

    # Getter and Setter for resultado_pk
    def get_resultado_pk(self):
        return self.resultado_pk

    def set_resultado_pk(self, resultado_pk):
        self.resultado_pk = resultado_pk


class Condicion:
    def __init__(self, id, tipo_condicion, nombre, cantidad, condicion_pk):
        self.id = id
        self.tipo_condicion = tipo_condicion
        self.nombre = nombre
        self.cantidad = cantidad
        self.condicion_pk = condicion_pk

    # Getter and Setter for id
    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    # Getter and Setter for tipo_condicion
    def get_tipo_condicion(self):
        return self.tipo_condicion

    def set_tipo_condicion(self, tipo_condicion):
        self.tipo_condicion = tipo_condicion

    # Getter and Setter for nombre
    def get_nombre(self):
        return self.nombre

    def set_nombre(self, nombre):
        self.nombre = nombre

    # Getter and Setter for cantidad
    def get_cantidad(self):
        return self.cantidad

    def set_cantidad(self, cantidad):
        self.cantidad = cantidad

    # Getter and Setter for condicion_pk
    def get_condicion_pk(self):
        return self.condicion_pk

    def set_condicion_pk(self, condicion_pk):
        self.condicion_pk = condicion_pk


class RecordMedico:
    def __init__(self, id, id_paciente):
        self.id = id
        self.id_paciente = id_paciente

    # Getter and Setter for id
    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    # Getter and Setter for id_paciente
    def get_id_paciente(self):
        return self.id_paciente

    def set_id_paciente(self, id_paciente):
        self.id_paciente = id_paciente


class VisitaCondicion:
    def __init__(self, id, id_condicion):
        self.id = id
        self.id_condicion = id_condicion

    # Getter and Setter for id
    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    # Getter and Setter for id_condicion
    def get_id_condicion(self):
        return self.id_condicion

    def set_id_condicion(self, id_condicion):
        self.id_condicion = id_condicion



class Visita:
    def __init__(self, id, num_visita, fecha, id_visita_condicion, id_resultado):
        self.id = id
        self.num_visita = num_visita
        self.fecha = fecha
        self.id_visita_condicion = id_visita_condicion
        self.id_resultado = id_resultado

    # Getter and Setter for id
    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    # Getter and Setter for num_visita
    def get_num_visita(self):
        return self.num_visita

    def set_num_visita(self, num_visita):
        self.num_visita = num_visita

    # Getter and Setter for fecha
    def get_fecha(self):
        return self.fecha

    def set_fecha(self, fecha):
        self.fecha = fecha

    # Getter and Setter for id_visita_condicion
    def get_id_visita_condicion(self):
        return self.id_visita_condicion

    def set_id_visita_condicion(self, id_visita_condicion):
        self.id_visita_condicion = id_visita_condicion

    # Getter and Setter for id_resultado
    def get_id_resultado(self):
        return self.id_resultado

    def set_id_resultado(self, id_resultado):
        self.id_resultado = id_resultado
