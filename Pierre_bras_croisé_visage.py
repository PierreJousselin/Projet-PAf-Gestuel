import json

import math


import cv2
import os


def afficher_frame(video_path, numero_frame,epaule_g,coude_g,poigné_g,epaule_d,coude_d,poigné_d,tete):

    #Pour Une vidéo affiche la frame de numéro= numero_frame et affiche l'image avec les keypoints :epaule etc

    # Ouvrir la vidéo
    video = cv2.VideoCapture(video_path)

    
    if not video.isOpened():
        print("Impossible d'ouvrir la vidéo.")
        return

    # Obtenir le nombre total de frames de la vidéo
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

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

def calculer_distance(point1, point2): #calcul la distance entre deux poitns
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




def get_data_front(annotation,num_interraction): 

    #récupere les informations  sur la partie frontal: renvoie (nombre_bras croisé/Total_frame , nombre_touché visage/Total_frame)
    # prend une annotation de format(t1,t2t,"confiance", et le numéro d'interraction)

    t1,t2,Tag=annotation #réucpere les timestam de toute les annoations de l'interraction
    num_fram_debut=round((t1*(10**-3)*50)/10)
    num_end_frame=round((t2*(10**-3)*50)/10)
    Nb_frame_croisé=0
    Nb_frame_visage=0
    total_frames=(num_end_frame-num_fram_debut)

    for i in range(num_fram_debut,num_end_frame): # pour chaque frame étudier
        # Chemin vers le fichier JSON à ouvrir
        chemin_fichier = "C://Users//Pierre//Documents//PAF//openpose//output"+str(num_interraction)+"wall//cam-table-wall_"+str(i * 10).zfill(12)+"_keypoints.json"
        chemin_video="C://Users//Pierre//Documents//PAF//PAF_2023//Dataset//Interactions//"+str(num_interraction)+"//cam-table-wall.mp4"
        total_frames=(num_end_frame-num_fram_debut)

        # Ouvrir le fichier JSON
        with open(chemin_fichier) as fichier:
            contenu = json.load(fichier)

        # Extraire la valeur associée à la clé "people"
        
        valeur_people = contenu["people"]

        if len(valeur_people)>1: #si les deux personnes sont détecter

            #Calcule pour la première personne

            keypointp1=valeur_people[0]["pose_keypoints_2d"]
            

            tete=keypointp1[0:3][0:2]
            epaule_g=keypointp1[2*3:3*3][0:2]
            coude_g=keypointp1[3*3:4*3][0:2]
            poigné_g=keypointp1[4*3:5*3][0:2]


            epaule_d=keypointp1[5*3:6*3][0:2]
            coude_d=keypointp1[6*3:7*3][0:2]
            poigné_d=keypointp1[7*3:8*3][0:2]

           #On vérifie que l'on a bien les données pour le calcul.
            if epaule_g!=[0,0] and coude_g!=[0,0] and poigné_g!=[0,0] and epaule_d!=[0,0] and coude_d!=[0,0] and poigné_d!=[0,0]:
                angle_g=calculer_angle(epaule_g,coude_g,poigné_g)
                angle_d=calculer_angle(epaule_d,coude_d,poigné_d)
                distance_pd_tete=calculer_distance(tete,poigné_d)
                distance_pg_tete=calculer_distance(tete,poigné_g)
                #print("frame number",str(i * 10),"angle_d:",angle_d,"nombre de personne",str(len(valeur_people)))

                if angle_g<120 and angle_d<120:
                    Nb_frame_croisé+=1
                    #afficher_frame(chemin_video,i*10,epaule_g,coude_g,poigné_g,epaule_d,coude_d,poigné_d,tete)
                    
                if distance_pg_tete<100 or distance_pd_tete<100:
                    Nb_frame_visage+=1
                    #afficher_frame(chemin_video,i*10,epaule_g,coude_g,poigné_g,epaule_d,coude_d,poigné_d,tete)
        

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
                distance_pd_tete=calculer_distance(tete,poigné_d)
                distance_pg_tete=calculer_distance(tete,poigné_g)
                #print("frame number",str(i * 10),"angle_d:",angle_d,"nombre de personne",str(len(valeur_people)))

                if angle_g<120 and angle_d<120:
                    Nb_frame_croisé+=1
                    #afficher_frame(chemin_video,i*10,epaule_g,coude_g,poigné_g,epaule_d,coude_d,poigné_d,tete)
                    
                if distance_pg_tete<100 or distance_pd_tete<100:
                    #afficher_frame(chemin_video,i*10,epaule_g,coude_g,poigné_g,epaule_d,coude_d,poigné_d,tete)
                    Nb_frame_visage+=1
        
         #Si il n' y avait que une personne détectée
        if len(valeur_people)==1:
            keypointp1=valeur_people[0]["pose_keypoints_2d"]
            

            tete=keypointp1[0:3][0:2]
            epaule_g=keypointp1[2*3:3*3][0:2]
            coude_g=keypointp1[3*3:4*3][0:2]
            poigné_g=keypointp1[4*3:5*3][0:2]


            epaule_d=keypointp1[5*3:6*3][0:2]
            coude_d=keypointp1[6*3:7*3][0:2]
            poigné_d=keypointp1[7*3:8*3][0:2]
            #print(epaule_d,epaule_g,coude_d,coude_g,tete,poigné_d,poigné_g)
            if epaule_g!=[0,0] and coude_g!=[0,0] and poigné_g!=[0,0] and epaule_d!=[0,0] and coude_d!=[0,0] and poigné_d!=[0,0]:
                angle_g=calculer_angle(epaule_g,coude_g,poigné_g)
                angle_d=calculer_angle(epaule_d,coude_d,poigné_d)
                distance_pd_tete=calculer_distance(tete,poigné_d)
                distance_pg_tete=calculer_distance(tete,poigné_g)
                #print("frame number",str(i * 10),"angle_d:",angle_d,"nombre de personne",str(len(valeur_people)))

                if angle_g<120 and angle_d<120:
                    Nb_frame_croisé+=1
                    #afficher_frame(chemin_video,i*10,epaule_g,coude_g,poigné_g,epaule_d,coude_d,poigné_d,tete)
                    
                if distance_pg_tete<100 or distance_pd_tete<100:
                    Nb_frame_visage+=1
                    #afficher_frame(chemin_video,i*10,epaule_g,coude_g,poigné_g,epaule_d,coude_d,poigné_d,tete)
        
    return(Nb_frame_croisé/total_frames,Nb_frame_visage/total_frames)
#print(get_data_front( (269048, 273413, 'Mistrusting'),27))