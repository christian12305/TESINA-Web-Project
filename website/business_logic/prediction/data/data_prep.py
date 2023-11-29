import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
import os 

#This class handles the data to be used to train the model
# it is designed for this specific heart disease dataset
# provided by the UCI Machine Learning Repository
# and the model chosen, GANDALF.
class ModelData:
    
    def __init__(self, data_path):
        #Read the data and store in pd.DataFrame
        self.__pd_data = pd.read_csv(data_path)
        self.__clean_data()
        
    #Method to clean the data
    def __clean_data(self):

        #Dropping missing value columns
        #they were eliminated because more than the 60% of rows were null
        #print((self.__pd_data['VasosPrincipales'].isnull().sum())/919)
        self.__pd_data = self.__pd_data.drop(['VasosPrincipales'], axis=1)
        self.__pd_data = self.__pd_data.drop(['Thal'], axis=1)

        #Filling missing values with the mean of the column
        #########################
        #ALTERNATE
        #self.__pd_data.Thal.fillna(3, inplace=True)
        #########################
        self.__pd_data['Slope(SegmentoST)'].fillna(self.__pd_data['Slope(SegmentoST)'].mean(), inplace=True)
        self.__pd_data['Oldpeak(SegmentoST)'].fillna(self.__pd_data['Oldpeak(SegmentoST)'].mean(), inplace=True)
        self.__pd_data.AnginaEjercicio.fillna(self.__pd_data.AnginaEjercicio.mean(), inplace=True)
        self.__pd_data.MaxRitmoCardiaco.fillna(self.__pd_data.MaxRitmoCardiaco.mean(), inplace=True)
        self.__pd_data.NivelesAzucarAyuna.fillna(self.__pd_data.NivelesAzucarAyuna.mean(), inplace=True)
        self.__pd_data.ECGDescanso.fillna(self.__pd_data.ECGDescanso.mean(), inplace=True)
        self.__pd_data.Colesterol.fillna(self.__pd_data.Colesterol.mean(), inplace=True)
        self.__pd_data.PresionArterialDescanso.fillna(self.__pd_data.PresionArterialDescanso.mean(), inplace=True)
        #Replacing the 0s in the Colesterol column
        self.__pd_data['Colesterol'] = self.__pd_data['Colesterol'].replace(0, self.__pd_data['Colesterol'].mean())

        #Data type changes
        self.__pd_data.PresionArterialDescanso = self.__pd_data.PresionArterialDescanso.astype('int64')
        self.__pd_data.Colesterol = self.__pd_data.Colesterol.astype('int64')
        self.__pd_data.NivelesAzucarAyuna = self.__pd_data.NivelesAzucarAyuna.astype('int64')
        self.__pd_data.ECGDescanso = self.__pd_data.ECGDescanso.astype('int64')
        self.__pd_data.MaxRitmoCardiaco = self.__pd_data.MaxRitmoCardiaco.astype('int64')
        self.__pd_data.AnginaEjercicio = self.__pd_data.AnginaEjercicio.astype('int64')
        self.__pd_data['Slope(SegmentoST)'] = self.__pd_data['Slope(SegmentoST)'].astype('int64')
        #########################
        #ALTERNATE
        #self.__pd_data.Thal = self.__pd_data.Thal.astype('int64')
        #########################
        
        #Scales down the different results [0,1,2,3,4] to [0,1]
        self.__pd_data['Prediccion'] = np.where(self.__pd_data.Prediccion> 0, 1, 0)
        self.__pd_data['Prediccion'] = self.__pd_data['Prediccion'].astype('category').cat.codes
        #########################

        #Eliminating duplicates
        self.__pd_data = self.__pd_data.drop_duplicates()
        #########################

    #Method to get the data provided after processing,
    # divided in train test and validation
    def get_data(self):
        #train_test_split is a method provided by scikit learn that
        # "Split arrays or matrices into random train and test subsets." 
        # (Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.)
        #train, test = train_test_split(self.__pd_data, test_size=0.2, random_state=24)
        #train, validation = train_test_split(train, test_size=0.2, random_state=24)
        #train, validation = train_test_split(self.__pd_data, test_size=0.2, random_state=24)
        #return (train, validation)
        return self.__pd_data
    
    #Method to get the features of the given data
    def get_features(self):
        #THESE ARE FOR HEART DISEASE, HENCE THE hd
        #hd_data_features = ['Edad', 'Sexo', 'Angina', 'PresionArterialDescanso', 'Colesterol', 'NivelesAzucarAyuna', 'ECGDescanso', 'MaxRitmoCardiaco', 'AnginaEjercicio', 'Oldpeak(SegmentoST)', 'Slope(SegmentoST)', 'Thal']
        hd_categorical_features = ['Sexo', 'Angina', 'NivelesAzucarAyuna', 'ECGDescanso', 'AnginaEjercicio', 'Slope(SegmentoST)' ]
        hd_continuous_features = ['Edad', 'PresionArterialDescanso', 'Colesterol', 'MaxRitmoCardiaco', 'Oldpeak(SegmentoST)']
        target = ['Prediccion']
        return (hd_categorical_features, hd_continuous_features, target)

    #Method to save the graphs of the data provided
    #Contains graphs for Exploratory Data Analysis (EDA)
    def save_graphs(self):
        
        #HeatMap
        plt.figure(figsize=(14,10))
        sns.heatmap(self.__pd_data.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)
        # Save the figure in the static directory  
        plt.savefig(os.path.join('website', 'static', 'images', 'heatmap.png')) 
        plt.close()

        ##########################

        #######Converting nominal variables#####
        CP_Dict = {1:'typical angina',2:'atypical angina',3:'non-anginal',4:'asymptomatic'}
        ECG_Dict = {0:'normal',1:'ST-T wave abnormality',2:'left ventricular hypertrophy'}
        #thal_Dict = {3:'normal',6:'fixed defect',7:'reversable defect'}

        self.__pd_data.replace({"Angina": CP_Dict},inplace=True)
        self.__pd_data.replace({"ECGDescanso": ECG_Dict},inplace=True)
        #self.__pd_data.replace({"Thal": thal_Dict},inplace=True)

        Sex_Dict = {1:'male',0:'female'}
        FS_Dict = {0:'under 120mgdl',1:'over 120mgdl'}
        exang_Dict = {0:'not induced',1:'induced'}
        slope_Dict = {1:'upsloping',2:'flat',3:'downsloping'}

        self.__pd_data.replace({"Sexo": Sex_Dict},inplace=True)
        self.__pd_data.replace({"NivelesAzucarAyuna": FS_Dict},inplace=True)
        self.__pd_data.replace({"AnginaEjercicio": exang_Dict},inplace=True)
        self.__pd_data.replace({"Slope(SegmentoST)": slope_Dict},inplace=True)


        ########## EDA #############

        #Target distribution
        f, axes = plt.subplots(1, 1, figsize=(4, 6))
        sns.countplot(ax=axes,x='Prediccion', data=self.__pd_data, palette=['green','orange'])
        axes.set_title("Target Distribution", fontsize=20)
        # Save the figure in the static directory  
        plt.savefig(os.path.join('website', 'static', 'images', 'Target_distribution.png')) 
        plt.close()

        #Categorical Binaries

        f, axes = plt.subplots(1, 3, figsize=(15, 5))

        sns.countplot(ax=axes[0],x='Sexo', data=self.__pd_data, palette=['green','orange'],hue="Prediccion")
        axes[0].set_title("Sexo", fontsize=20)

        sns.countplot(ax=axes[1],x='NivelesAzucarAyuna', data=self.__pd_data, palette=['green','orange'],hue="Prediccion")
        axes[1].set_title("NivelesAzucarAyuna", fontsize=20)

        sns.countplot(ax=axes[2],x='AnginaEjercicio', data=self.__pd_data, palette=['green','orange'],hue="Prediccion")
        plt.title("AnginaEjercicio", fontsize=20)

        # Save the figure in the static directory  
        plt.savefig(os.path.join('website', 'static', 'images', 'SNA.png')) 
        plt.close()

        #Categorical Variables

        plt.figure(figsize=(12,5))
        sns.countplot(x='Angina', data=self.__pd_data, palette=['green','orange'],hue="Prediccion")
        plt.title("Angina", fontsize=20)
        # Save the figure in the static directory  
        plt.savefig(os.path.join('website', 'static', 'images', 'Angina.png')) 
        plt.close()

        plt.figure(figsize=(12,5))
        sns.countplot(x='Slope(SegmentoST)', data=self.__pd_data, palette=['green','orange'],hue="Prediccion")
        plt.title("Slope(SegmentoST)", fontsize=20)
        # Save the figure in the static directory  
        plt.savefig(os.path.join('website', 'static', 'images', 'Slope(SegmentoST).png')) 
        plt.close()

        plt.figure(figsize=(12,5))
        ax=sns.countplot(x='ECGDescanso', data=self.__pd_data, palette=['green','orange'],hue="Prediccion")
        plt.title("ECGDescanso", fontsize=20)
        # Save the figure in the static directory  
        plt.savefig(os.path.join('website', 'static', 'images', 'ECGDescanso.png')) 
        plt.close()

        #plt.figure(figsize=(12,5))
        #sns.countplot(x='Thal', data=self.__pd_data, palette=['green','orange'],hue="Prediccion")
        #plt.title("Thal", fontsize=20)

        data_disease = self.__pd_data[self.__pd_data["Prediccion"] == 1]
        data_normal = self.__pd_data[self.__pd_data["Prediccion"] == 0]

        plt.figure(figsize=(8,5))
        sns.distplot(data_normal["Edad"], bins=24, color='g')
        sns.distplot(data_disease["Edad"], bins=24, color='r')
        plt.title("Distribucion y densidad por Edad",fontsize=20)
        plt.xlabel("Edad",fontsize=15)
        # Save the figure in the static directory  
        plt.savefig(os.path.join('website', 'static', 'images', 'dist_densidad_edad.png')) 
        plt.close()

        plt.figure(figsize=(8,5))
        sns.distplot(data_normal["Colesterol"], bins=24, color='g')
        sns.distplot(data_disease["Colesterol"], bins=24, color='r')
        plt.title("Distribucion y densidad por colesterol",fontsize=20)
        plt.xlabel("Colesterol",fontsize=15)
        # Save the figure in the static directory  
        plt.savefig(os.path.join('website', 'static', 'images', 'dist_densidad_chol.png')) 
        plt.close()

        plt.figure(figsize=(8,5))
        sns.distplot(data_normal["PresionArterialDescanso"], bins=24, color='g')
        sns.distplot(data_disease["PresionArterialDescanso"], bins=24, color='r')
        plt.title("Distribucion y densidad por Presion Arterial en Descanso",fontsize=20)
        plt.xlabel("PresionArterialDescanso",fontsize=15)
        # Save the figure in the static directory  
        plt.savefig(os.path.join('website', 'static', 'images', 'dist_densidad_rbp.png')) 
        plt.close()

        plt.figure(figsize=(8,5))
        sns.distplot(data_normal["MaxRitmoCardiaco"], bins=24, color='g')
        sns.distplot(data_disease["MaxRitmoCardiaco"], bins=24, color='r')
        plt.title("Distribucion y densidad por MaxRitmoCardiaco",fontsize=20)
        plt.xlabel("MaxRitmoCardiaco",fontsize=15)
        # Save the figure in the static directory  
        plt.savefig(os.path.join('website', 'static', 'images', 'dist_densidad_maxHR.png')) 
        plt.close()

        plt.figure(figsize=(8,5))
        sns.distplot(data_normal["Oldpeak(SegmentoST)"], bins=24, color='g')
        sns.distplot(data_disease["Oldpeak(SegmentoST)"], bins=24, color='r')
        plt.title("Distribucion y densidad por Oldpeak(SegmentoST)",fontsize=20)
        plt.xlabel("oldpeak",fontsize=15)
        # Save the figure in the static directory  
        plt.savefig(os.path.join('website', 'static', 'images', 'dist_densidad_oldpeak.png')) 
        plt.close()

        plt.figure(figsize=(9, 7))
        plt.scatter(data_disease["Edad"],
                    data_disease["MaxRitmoCardiaco"],
                    c="salmon")
        plt.scatter(data_normal["Edad"],
                    data_normal["MaxRitmoCardiaco"],
                    c="lightblue")
        plt.title("Heart Disease in function of Age and Max Heart Rate")
        plt.xlabel("Age")
        plt.ylabel("Max Heart Rate")
        plt.legend(["Disease", "No Disease"])
        # Save the figure in the static directory  
        plt.savefig(os.path.join('website', 'static', 'images', 'function_age_maxHR.png')) 
        plt.close()

        #Correlation matrices
        #sns.pairplot(data=self.__pd_data)