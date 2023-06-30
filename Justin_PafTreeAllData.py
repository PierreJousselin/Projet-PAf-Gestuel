#ce programme crée une random forest à partir de toutes les données avec earlyfusion

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, recall_score,f1_score
from joblib import dump
import pympi
import numpy as np
import os
import pandas as pd



# n_exclu est l'interaction qui servira pour tester le modèle créé avec les 9 autres
path="C:\\Users\\justi\\Data_geste_Justin\\"
n_exclu="15"

#données d'entraînement et de test
X_train=[]
Y_train=[]
X_test=[]
Y_test=[]

#liste des interactions et liste des dossiers de données par modalité
listInter=['9','10','12','15','18','19','24','26','27','30']
listData=["Data_geste","Data_distance","Data_geste_2","Data_prosodie","data_semantique","data_AUs","Data_emotion"]

for n in listInter:
    if n!=n_exclu:
        #nbre est le nombre de segments dans une interaction
        nbre=os.listdir(path+n)

        #on récupère les annotations de confiance pour Y_train
        eaf = pympi.Elan.Eaf("C:\\Users\\justi\\OneDrive\\Documents\\PAF_2023\\Dataset\\Interactions\\"+n+"\\"+n+".eaf")
        trust = sorted(eaf.get_annotation_data_for_tier('Trust'))

        #pour chaque segment de l'interaction n
        for i in range(len(nbre)):
            l=[]
            #pour chaque modalité
            for j in listData:
                #on load les données des fichiers .npy
                a=np.load("C:\\Users\\justi\\"+j+"\\"+n+"\\segment_"+str(i)+".npy")
                #on règle un problème de formatage pour Data_prosodie et Data_distance, 
                # qui donnent des arrays d'arrays au lieu d'arrays 1-D
                if (j=="Data_prosodie" or j=="Data_distance"):
                    a=np.concatenate(a)
                l.append(a)
            l=np.concatenate(l)

            #on stocke les données dans X_train et Y_train
            X_train.append(l)
            Y_train.append(trust[i][2])

            #On triple le nombre de mistrusting
            if trust[i][2]=='Mistrusting':
                X_train.append(l)
                Y_train.append(trust[i][2])
                X_train.append(l)
                Y_train.append(trust[i][2])


#on refait la procédure précédente pour l'interaction test et pour constituer X_test et Y_test
nbre=os.listdir(path+n_exclu)
eaf = pympi.Elan.Eaf("C:\\Users\\justi\\OneDrive\\Documents\\PAF_2023\\Dataset\\Interactions\\"+n_exclu+"\\"+n_exclu+".eaf")
trust = sorted(eaf.get_annotation_data_for_tier('Trust'))
for i in range(len(nbre)):
    l=[]
    for j in listData:
        a=np.load("C:\\Users\\justi\\"+j+"\\"+n_exclu+"\\segment_"+str(i)+".npy")
        if j=="Data_prosodie" or j=="Data_distance":
            a=np.concatenate(a)
        l.append(a)
    X_test.append(np.concatenate(l))
    Y_test.append(trust[i][2])

#on remplace les strings par du binaire
Y_train=MultiLabelBinarizer().fit_transform(Y_train)
Y_test=MultiLabelBinarizer().fit_transform(Y_test)

#on crée le pipeline
pipelineRFC = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=1))

#on crée la grille de paramètres
param_grid_rfc = [{
    'randomforestclassifier__n_estimators': [90,100,115,130], #100 sur [90,100,115,130]
    'randomforestclassifier__max_depth':[2,5,8,11,14,17,20], #17 sur [2,5,8,11,14,17,20]; 15 sur [15,16,17,18,19]
    'randomforestclassifier__max_features':["sqrt","log2"], #log2 sur ["sqrt","log2"]
    'randomforestclassifier__min_samples_leaf':[2,4,6,8,10], #2 sur [2,4,6,8,10]; 2 sur [1,2,3]
    'randomforestclassifier__criterion':["gini","entropy"] #entropy sur ["gini","entropy"]

}]



# Create an instance of GridSearch Cross-validation estimator
gsRFC = GridSearchCV(estimator=pipelineRFC,
                     param_grid = param_grid_rfc,
                     scoring='accuracy',
                     cv=10,
                     refit=True,
                     n_jobs=-1)

#on train
gsRFC = gsRFC.fit(X_train, Y_train)

# On affiche le meilleur score
print(gsRFC.best_score_)

# On affiche les paramètres du meilleur modèle
print(gsRFC.best_params_)

#on prend le meilleur modèle et on le save dans le fichier 'RFCtest.joblib'
clfRFC = gsRFC.best_estimator_

dump(clfRFC, 'RFCtest.joblib')



# on teste avec X_test et on print les scores des différentes mesures pertinentes
Y_pred=clfRFC.predict(X_test)

accuracy = accuracy_score(Y_pred, Y_test)
recall=recall_score(Y_pred, Y_test,average='macro')
f1=f1_score(Y_pred, Y_test,average='macro')
print("recall:"+str(recall))
print("accuracy"+str(accuracy))
print("f1:"+str(f1))

