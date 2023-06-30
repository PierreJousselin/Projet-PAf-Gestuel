#ce programme load un model et le teste sur les données d'une interaction donnée

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, recall_score,f1_score
from joblib import load
import pympi
import numpy as np
import os

path="C:\\Users\\justi\\Data_geste_Justin\\"
#n_exclu numéro de l'interaction à tester (donc l'interaction exclue du training)
n_exclu="9"

X_test=[]
Y_test=[]

#listInter=['9','10','12','15','18','19','24','26','27','30'] pense-bête

#list_Data liste des dossiers de données des modalités
listData=["Data_geste","Data_distance","Data_geste_2","Data_prosodie","data_semantique","data_AUs","Data_emotion"]


#nbre est le nombre de segments dans l'interaction n_exclu
nbre=os.listdir(path+n_exclu)
#on récupère les annotations de confiance pour Y_test
eaf = pympi.Elan.Eaf("C:\\Users\\justi\\OneDrive\\Documents\\PAF_2023\\Dataset\\Interactions\\"+n_exclu+"\\"+n_exclu+".eaf")
trust = sorted(eaf.get_annotation_data_for_tier('Trust'))

#pour chaque segment de l'interaction n_exclu
for i in range(len(nbre)):
    l=[]
    #pour chaque modalité
    for j in listData:
        #on load les données des fichiers .npy
        a=np.load("C:\\Users\\justi\\"+j+"\\"+n_exclu+"\\segment_"+str(i)+".npy")
        if j=="Data_prosodie" or j=="Data_distance":
            a=np.concatenate(a)
        l.append(a)
    #on stocke les données dans X_test et Y_test
    X_test.append(np.concatenate(l))
    Y_test.append(trust[i][2])

#on remplace les strings par du binaire
Y_test=MultiLabelBinarizer().fit_transform(Y_test)

#on load le modèle
ModelRfc=load('RFCtest.joblib') 


# On teste avec l'interaction exclue et on affiche les résultats
Y_pred=ModelRfc.predict(X_test)

accuracy = accuracy_score(Y_pred, Y_test)
recall=recall_score(Y_pred, Y_test,average='macro')
f1=f1_score(Y_pred, Y_test,average='macro')
print("recall:"+str(recall))
print("accuracy"+str(accuracy))
print("f1:"+str(f1))