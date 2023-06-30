# ce programme crée une random forest en se basant uniquement sur les données gestuelles²


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

# n_exclu est l'interaction qui servira pour tester le modèle créé avec les 9 autres
n_exclu="10"

path="C:\\Users\\justi\\Data_geste_Justin\\"

#données d'entraînement
X_train=[]
Y_train=[]

listInter=['9','10','12','15','18','19','24','26','27','30']
for n in listInter:
    if n!=n_exclu:
        #nbre est le nombre de segments dans une interaction
        nbre=os.listdir(path+n)

        #on récupère les annotations de confiance pour Y_train
        eaf = pympi.Elan.Eaf("C:\\Users\\justi\\OneDrive\\Documents\\PAF_2023\\Dataset\\Interactions\\"+n+"\\"+n+".eaf")
        trust = sorted(eaf.get_annotation_data_for_tier('Trust'))

        #pour chaque segment de l'interaction n
        for i in range(len(nbre)):
            
            #on load les données des fichiers .npy
            a=np.load("C:\\Users\\justi\\Data_geste_2\\"+n+"\\segment_"+str(i)+".npy")
            b=np.load("C:\\Users\\justi\\Data_geste\\"+n+"\\segment_"+str(i)+".npy")
            
            #on stocke les données dans X_train et Y_train
            X_train.append(np.concatenate([a,b]))
            Y_train.append(trust[i][2])

            #on triple les données "mistrusting"
            if trust[i][2]=='Mistrusting':
                for q in range(2): #for q in range(4) pour égaliser totalement les classes
                    X_train.append(np.concatenate([a,b]))
                    Y_train.append(trust[i][2])
            
            #if trust[i][2]=='Trusting': #pour égaliser totalement les classes
                #X_train.append(np.concatenate([a,b]))
                #Y_train.append(trust[i][2])



#on refait la procédure précédente pour l'interaction test et pour constituer X_test et Y_test
X_test=[]
Y_test=[]
nbre=os.listdir(path+n_exclu)
eaf = pympi.Elan.Eaf("C:\\Users\\justi\\OneDrive\\Documents\\PAF_2023\\Dataset\\Interactions\\"+n_exclu+"\\"+n_exclu+".eaf")
trust = sorted(eaf.get_annotation_data_for_tier('Trust'))
for i in range(len(nbre)):
    a=np.load("C:\\Users\\justi\\Data_geste_2\\"+n_exclu+"\\segment_"+str(i)+".npy")
    b=np.load("C:\\Users\\justi\\Data_geste\\"+n_exclu+"\\segment_"+str(i)+".npy")
    X_test.append(np.concatenate([a,b]))
    Y_test.append(trust[i][2])

#on remplace les strings par du binaire
Y_train=MultiLabelBinarizer().fit_transform(Y_train)
Y_test=MultiLabelBinarizer().fit_transform(Y_test)

#on crée le pipeline
pipelineRFC = make_pipeline(RandomForestClassifier(criterion='gini', random_state=1))

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

#On train
gsRFC = gsRFC.fit(X_train, Y_train)

# On affiche le meilleur score
print(gsRFC.best_score_)

# On affiche les paramètres du meilleur modèle
print(gsRFC.best_params_)

#on prend le meilleur modèle et on le save dans le fichier 'RFCgeste.joblib'
clfRFC = gsRFC.best_estimator_
dump(clfRFC, 'RFCgeste.joblib')


# On teste avec l'interaction exclue et on affiche les résultats
Y_pred=clfRFC.predict(X_test)

accuracy = accuracy_score(Y_pred, Y_test)
recall=recall_score(Y_pred, Y_test,average='macro')
f1=f1_score(Y_pred, Y_test,average='macro')
print("accuracy"+str(accuracy))
print("recall:"+str(recall))
print("f1:"+str(f1))
