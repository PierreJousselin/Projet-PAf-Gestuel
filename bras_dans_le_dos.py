import json

import math


import cv2
import os
#os.chdir("C://Users//Pierre//Documents//PAF//PAF_2023//Dataset//Interactions//9")
def afficher_frame(video_path, numero_frame,epaule_g,coude_g,poigné_g,epaule_d,coude_d,poigné_d,tete):
    # Ouvrir la vidéo
    video = cv2.VideoCapture(video_path)

    # Vérifier si la vidéo a été ouverte avec succès
    if not video.isOpened():
        print("Impossible d'ouvrir la vidéo.")
        return

    # Obtenir le nombre total de frames de la vidéo
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    # Vérifier si le numéro de frame est valide
    if numero_frame < 0 or numero_frame >= total_frames:
        print("Numéro de frame invalide.")
        video.release()
        return

    # Positionner la vidéo au numéro de frame spécifié
    video.set(cv2.CAP_PROP_POS_FRAMES, numero_frame)

    # Lire la frame correspondante
    ret, frame = video.read()

    # Vérifier si la lecture de la frame a réussi
    if not ret:
        print("Erreur lors de la lecture de la frame.")
        video.release()
        return

    # Afficher la frame
    
    cv2.circle(frame, (int(epaule_g[0]),int(epaule_g[1])), 5, (0, 0, 255))
    cv2.circle(frame, (int(coude_g[0]),int(coude_g[1])), 5, (0, 255, 0))
    cv2.circle(frame, (int(poigné_g[0]),int(poigné_g[1])), 5, (255, 0, 0))
    cv2.circle(frame, (int(epaule_d[0]),int(epaule_d[1])), 5, (0, 0, 255))
    cv2.circle(frame, (int(coude_d[0]),int(coude_d[1])), 5, (0, 255, 0))
    cv2.circle(frame, (int(poigné_d[0]),int(poigné_d[1])), 5, (255, 0, 0))
    cv2.circle(frame, (int(tete[0]),int(tete[1])), 5, (0, 0, 255))
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)

    # Libérer les ressources
    video.release()
    cv2.destroyAllWindows()


import math

def calculer_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculer_angle(A, B, C):

    # Vecteurs AB et BC
    vecteur_AB = (B[0] - A[0], B[1] - A[1])
    vecteur_BC = (C[0] - B[0], C[1] - B[1])

    # Produit scalaire entre les vecteurs AB et BC
    produit_scalaire = vecteur_AB[0] * vecteur_BC[0] + vecteur_AB[1] * vecteur_BC[1]

    # Normes des vecteurs AB et BC
    norme_AB = math.sqrt(vecteur_AB[0] ** 2 + vecteur_AB[1] ** 2)
    norme_BC = math.sqrt(vecteur_BC[0] ** 2 + vecteur_BC[1] ** 2)

    # Calcul de l'angle en radians
    angle_rad = math.acos(produit_scalaire / (norme_AB * norme_BC))

    # Conversion de l'angle en degrés
    angle_deg = math.degrees(angle_rad)

    return (abs(180-angle_deg))



def get_data_dos(annotation,num_interraction):
    t1,t2,Tag=annotation # réupérationdes timestamp
    num_fram_debut=round((t1*(10**-3)*50)/10) #calcul des frames
    num_end_frame=round((t2*(10**-3)*50)/10)
    
    Nb_frame_dos=0
    total_frames=(num_end_frame-num_fram_debut)
    for i in range(num_fram_debut,num_end_frame):

        # Chemin vers le fichier JSON à ouvrir
        chemin_fichier = "C://Users//Pierre//Documents//PAF//openpose//output"+str(num_interraction)+"rear//cam-rear_"+str(i * 10).zfill(12)+"_keypoints.json"

        # Ouvrir le fichier JSON
        with open(chemin_fichier) as fichier:
            contenu = json.load(fichier)

        # Extraire la valeur associée à la clé "people"
        
        valeur_people = contenu["people"]
        
        #Si les deux personnes sont détecter
        if len(valeur_people)>1:
            #Calcule pour la première
            keypointp1=valeur_people[0]["pose_keypoints_2d"]
           

            tete=keypointp1[0:3][0:2]
            epaule_g=keypointp1[2*3:3*3][0:2]
            coude_g=keypointp1[3*3:4*3][0:2]
            poigné_g=keypointp1[4*3:5*3][0:2]


            epaule_d=keypointp1[5*3:6*3][0:2]
            coude_d=keypointp1[6*3:7*3][0:2]
            poigné_d=keypointp1[7*3:8*3][0:2]
            
            #on vérifie que l'on a bien les données pour calculer.
            if epaule_g!=[0,0] and coude_g!=[0,0] and poigné_g!=[0,0] and epaule_d!=[0,0] and coude_d!=[0,0] and poigné_d!=[0,0]:
                angle_g=calculer_angle(epaule_g,coude_g,poigné_g)
                angle_d=calculer_angle(epaule_d,coude_d,poigné_d)
                distance_pd_pg=calculer_distance(poigné_d,poigné_g)
                

                 #Formule pour détecter les bras dans le dos (voir rapport)   
                if distance_pd_pg<40 and (poigné_d[1]>coude_d[1]+17 and poigné_g[1]>coude_g[1]+17) and abs(poigné_g[0]-poigné_d[0])<40 and (poigné_d[0]>coude_d[0]-10 and poigné_g[0]<coude_g[0]+10) :
                    #print("frame number",str(i * 10),"angle_d:",angle_d,"nombre de personne",str(len(valeur_people)),"nu")
                    Nb_frame_dos+=1
                    #afficher_frame("C://Users//Pierre//Documents//PAF//PAF_2023//Dataset//Interactions//"+str(num_interraction)+"//cam-rear.mp4",i*10,epaule_g,coude_g,poigné_g,epaule_d,coude_d,poigné_d,tete)


            #Calcule pour la deuxième personne
            keypointp1=valeur_people[1]["pose_keypoints_2d"]

            tete=keypointp1[0:3][0:2]
            epaule_g=keypointp1[2*3:3*3][0:2]
            coude_g=keypointp1[3*3:4*3][0:2]
            poigné_g=keypointp1[4*3:5*3][0:2]


            epaule_d=keypointp1[5*3:6*3][0:2]
            coude_d=keypointp1[6*3:7*3][0:2]
            poigné_d=keypointp1[7*3:8*3][0:2]
            
            if epaule_g!=[0,0] and coude_g!=[0,0] and poigné_g!=[0,0] and epaule_d!=[0,0] and coude_d!=[0,0] and poigné_d!=[0,0]:
                angle_g=calculer_angle(epaule_g,coude_g,poigné_g)
                angle_d=calculer_angle(epaule_d,coude_d,poigné_d)
                distance_pd_pg=calculer_distance(poigné_d,poigné_g)
               

                    
                if distance_pd_pg<40 and (poigné_d[1]>coude_d[1]+17 and poigné_g[1]>coude_g[1]+17) and abs(poigné_g[0]-poigné_d[0])<40 and (poigné_d[0]>coude_d[0]-10 and poigné_g[0]<coude_g[0]+10) :
                    #print("frame number",str(i * 10),"angle_d:",angle_d,"nombre de personne",str(len(valeur_people)),"yo")
                    Nb_frame_dos+=1
                    #afficher_frame("C://Users//Pierre//Documents//PAF//PAF_2023//Dataset//Interactions//"+str(num_interraction)+"//cam-rear.mp4",i*10,epaule_g,coude_g,poigné_g,epaule_d,coude_d,poigné_d,tete)

    if len(valeur_people)==1:
            #Si il n'y a qu'une personne
            keypointp1=valeur_people[0]["pose_keypoints_2d"]
         

            tete=keypointp1[0:3][0:2]
            epaule_g=keypointp1[2*3:3*3][0:2]
            coude_g=keypointp1[3*3:4*3][0:2]
            poigné_g=keypointp1[4*3:5*3][0:2]


            epaule_d=keypointp1[5*3:6*3][0:2]
            coude_d=keypointp1[6*3:7*3][0:2]
            poigné_d=keypointp1[7*3:8*3][0:2]
            
            if epaule_g!=[0,0] and coude_g!=[0,0] and poigné_g!=[0,0] and epaule_d!=[0,0] and coude_d!=[0,0] and poigné_d!=[0,0]:
                angle_g=calculer_angle(epaule_g,coude_g,poigné_g)
                angle_d=calculer_angle(epaule_d,coude_d,poigné_d)
                distance_pd_pg=calculer_distance(poigné_d,poigné_g)
                

                    
                if distance_pd_pg<60 and (poigné_d[1]>coude_d[1]+17 and poigné_g[1]>coude_g[1]+17) and abs(poigné_g[0]-poigné_d[0])<40 and (poigné_d[0]>coude_d[0]-10 and poigné_g[0]<coude_g[0]+10) :
                    #print("frame number",str(i * 10),"angle_d:",angle_d,"nombre de personne",str(len(valeur_people)),"couc")
                    Nb_frame_dos+=1
                    #afficher_frame("C://Users//Pierre//Documents//PAF//PAF_2023//Dataset//Interactions//"+str(num_interraction)+"//cam-rear.mp4",i*10,epaule_g,coude_g,poigné_g,epaule_d,coude_d,poigné_d,tete)


    return(Nb_frame_dos/total_frames)


