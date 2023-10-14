from enum import Enum

#Enumerator for CondicionType
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