class User:
    def __init__(self, id, primer_nombre, inicial, apellido_paterno, correo_electronico, contraseña, rol):
        self.__set_id(id)
        self.__set_primer_nombre(primer_nombre)
        self.__set_inicial(inicial)
        self.__set_apellido_paterno(apellido_paterno)
        self.__set_correo_electronico(correo_electronico)
        self.__set_contraseña(contraseña)
        self.__set_rol(rol)

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

    # Getter and Setter for role
    def get_rol(self):
        if(self.rol == 1):
            return 'user'
        else:
            return 'admin'

    def __set_rol(self, rol):
        self.rol = rol