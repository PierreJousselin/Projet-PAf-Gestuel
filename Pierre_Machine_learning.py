
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score,f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import History 
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pympi
def remove_duplicates(X, Y):
    unique_X = []
    unique_Y = []
    indices_seen = set()

    for i, x in enumerate(X):
        if tuple(x) not in indices_seen:
            unique_X.append(x)
            unique_Y.append(Y[i])
            indices_seen.add(tuple(x))

    return unique_X, unique_Y
def somme_couples(liste_couples):
    somme = [0, 0, 0]  # Initialisation du couple de somme à [0, 0, 0]
    for couple in liste_couples:
        somme[0] += couple[0]  # Somme des premières valeurs des couples
        somme[1] += couple[1]  # Somme des deuxièmes valeurs des couples
        somme[2] += couple[2]  # Somme des troisièmes valeurs des couples
    return tuple(somme)  # Conversion de la liste en un couple de valeurs

def comparer_elements(X, Y):
    res=0
    for element_x , elm_Y in zip(X,Y):
        couple_converti = list([1 if val == max(element_x) else 0 for val in element_x])
        if couple_converti == elm_Y:
            res+=1
        
    return res/len(Y)



def extract_data():
    X=[]
    Y=[]
    Y_tag=[]
    for i in [9,10,12,15,18,19,24,26,27,30]:
        file ="C://Users//Pierre//Documents//PAF//PAF_2023//Dataset//Interactions//"+str(i)+"//"+str(i)+".eaf"
        eaf = pympi.Elan.Eaf(file)
        annots = sorted(eaf.get_annotation_data_for_tier('Trust'))
        
        repertoire_mouvement="C://Users//Pierre//Documents//PAF//Data_mouvement//"+str(i)
        repertoire_geste="C://Users//Pierre//Documents//PAF//Data_geste//"+str(i)
        repertoire_emotion="C://Users//Pierre//Documents//PAF//Data_emotion//"+str(i)
        repertoire_au="C://Users//Pierre//Documents//PAF//Data_AUs//"+str(i)
        repertoire_prosodie="C://Users//Pierre//Documents//PAF//Data_prosodie//"+str(i)
        repertoire_semantique="C://Users//Pierre//Documents//PAF//Data_semantique//"+str(i)
        repertoire_distance="C://Users//Pierre//Documents//PAF//Data_distance//"+str(i)
        data_m=[] #liste pour l'Interraction dont chaque élement est le vecteur de données des mouvements pour le segment
        data_g=[] 
        data_au=[]
        data_e=[]
        data_p=[]
        data_s=[]
        data_d=[]

        for num_segment in range(len(os.listdir(repertoire_mouvement))):
            
            filen_name=repertoire_mouvement+"//segment_"+str(num_segment)+".npy"
            data_mouvement_str=list(np.load(filen_name))
            data_mouvement=[float(x) for x in data_mouvement_str[:]]
            data_m.append(data_mouvement)
        
        
        for num_segment in range(len(os.listdir(repertoire_geste))):
            filen_name=repertoire_geste+"//segment_"+str(num_segment)+".npy"
            data_geste=list(np.load(filen_name))
            data_g.append(data_geste)
            #print(data_geste)
            
        for num_segment in range(len(os.listdir(repertoire_geste))):
            filen_name=repertoire_emotion+"//segment_"+str(num_segment)+".npy"
            data_emotion=list(np.load(filen_name))
            data_e.append(data_emotion)
            #print(data_emotion)

        for num_segment in range(len(os.listdir(repertoire_geste))):
            filen_name=repertoire_semantique+"//segment_"+str(num_segment)+".npy"
            data_semantique=list(np.load(filen_name))
            data_s.append(data_semantique)
            #print(data_semantique)

        for num_segment in range(len(os.listdir(repertoire_geste))):
            filen_name=repertoire_prosodie+"//segment_"+str(num_segment)+".npy"
            data_prosodie=np.load(filen_name)[0]
            #print(len(data_prosodie))
            data_p.append(data_prosodie)
        
        for num_segment in range(len(os.listdir(repertoire_geste))):
            filen_name=repertoire_distance+"//segment_"+str(num_segment)+".npy"
            data_distance_brut=np.load(filen_name)
            data_distance=list(data_distance_brut.flatten())
            data_d.append(data_distance)
            

        for num_segment in range(len(os.listdir(repertoire_geste))):
            filen_name=repertoire_au+"//segment_"+str(num_segment)+".npy"
            data_au_brut=list(np.load(filen_name))
            data_au.append(data_au_brut)
        data_yi_tag  =[]
        data_xi=[]
        data_yi=[]
        
        if len(data_g)!=len(data_m):
            print("pb", len(data_g),len(data_m))

        else:
            for k in range(len(data_g)):
               
                data_xi.append(data_g[k]+data_au[k]+data_e[k]+data_s[k]+data_m[k]) #Concatenate les données
                
                tag =annots[k][2]
                
                if tag=="Neutral":
                    y=[0,1,0]
                if tag=="Trusting":
                    y=[0,0,1]
                if tag=="Mistrusting":
                    y=[1,0,0]
                
                data_yi.append(y)
                data_yi_tag.append(tag)
                if tag=="Mistrusting"  and False: #dupliquer les données de type Mistrusting
                    data_xi.append(data_g[k]+data_au[k]+data_e[k]+data_s[k]+data_m[k])
                    data_yi.append(y)
                    data_yi_tag.append(tag)
                    data_xi.append(data_g[k]+data_au[k]+data_e[k]+data_s[k]+data_m[k])
                    data_yi.append(y)
                    data_yi_tag.append(tag)
                    
                if tag=="Trusting" and False: #dupliquer les segment de confiance
                    data_xi.append(data_g[k]+data_m[k])
                    data_yi.append(y)
                    data_yi_tag.append(tag)
                    

        X.append(data_xi)
        Y.append(data_yi)
        Y_tag.append(data_yi_tag)
        
    
    
    return(X,Y,Y_tag)
    

def learn(X,Y,Y_tag,cross_validation=False,draw_graph_each = False):

    Total_prdeiction=[] # liste qui conteitn toute les prédiction
    Total_real_classes=[]#liste qui contient toutes les vrais classes pour chauqe prédictions
    for i in range(0,10): # chois l'interraction qui est en Test
        X_test=X[i]
        Y_test=Y[i]
        X_train=[]
        Y_train=[]
        Y_train_tag=[]
        Y_test_tag=Y_tag[i]
        for j in range(0,10): #j'ajoute toutes les données sauf celle de l'interraction chosis pour le test
            if j!=i:
                X_train+= X[j]
                Y_train+= Y[j]
                Y_train_tag+=Y_tag[j]
                
        X_test,Y_test_tag=remove_duplicates(X_test,Y_test_tag)
        
        model =RandomForestClassifier()
        param_grid = {
        'n_estimators': [100, 200,300,250,150],
        'max_depth': [3,5,8, 10, 15,20],
        'max_features' : ['log2','sqrt'],
        'min_samples_leaf': [4, 8, 16,25,32,48]
        }
        cv = StratifiedKFold(n_splits=5)
    
        

        if cross_validation:
        
            grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring='f1_macro')
            grid_search.fit(X_train, Y_train_tag)
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            print(i, best_params, best_score)
            print(best_model.score(X_test, Y_test_tag))
            predictions = best_model.predict(X_test)
            Total_prdeiction += list(predictions)
            Total_real_classes += Y_test_tag
            print(accuracy_score(Y_test_tag, predictions), recall_score(Y_test_tag, predictions,average='macro'),f1_score(Y_test_tag, predictions,average='macro'))

            if draw_graph_each:

                confusion_mat = confusion_matrix(Y_test_tag, predictions)

                # Obtenir les classes réelles
                classes = ['Mistrusting', 'Neutral', 'Trusting']

                # Créer le graphique
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
                plt.xlabel('Classe prédite')
                plt.ylabel('Classe réelle')
                plt.title('Matrice de confusion sur interraction'+str(i))
                plt.show()



    confusion_mat = confusion_matrix(Total_real_classes, Total_prdeiction)

    # Obtenir les classes réelles
    classes = ['Mistrusting', 'Neutral', 'Trusting']

    # Créer le graphique
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Classe prédite')
    plt.ylabel('Classe réelle')
    plt.title('Matrice de confusion sur gestuelle')
    plt.show()

    if False: #Pour tester le modèle de neuronnes
    
                    
        model = Sequential()
        history = History()

        model.add(Dense(128, activation='relu', input_shape=(39,)))  # Couche d'entrée
        model.add(Dropout(0.2))  # Ajout d'une couche de Dropout avec un taux de 0.2

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(3, activation='softmax'))

        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    
        model.fit(X_train, Y_train, epochs=160, batch_size=30, verbose=False,callbacks=[history])
        accuracies = history.history['accuracy']

        # Évaluation du modèle sur les données de test
        loss, accuracy = model.evaluate(X_test, Y_test)
        
        losses = history.history['loss']

        # Affichage du graphique de l'exactitude
        plt.plot(range(1, len(accuracies) + 1), accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Epochs')
        plt.show()

        print(loss,accuracy, comparer_elements(model.predict(X_test),Y_test))
        print(somme_couples(Y_test))

X,Y, Y_tag =extract_data()
print(len(X[0][0]))
learn(X,Y,Y_tag,cross_validation=True)

