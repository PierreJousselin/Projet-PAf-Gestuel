import csv
import pympi
from bs4 import BeautifulSoup
import lxml
import numpy as np

#listInter=['9','10','12','15','18','19','24','26','27','30'] pense-bête pour les interactions
n="30" #numéro de l'interaction dont on veut extraire les données

#on lit les timestamp pour elan et le clapperboard
with open("C:\\Users\\justi\\OneDrive\\Documents\\PAF_2023\\Dataset\\Interactions\\"+n+"\\timestamp-elan-begin.txt") as f:
    init_elan = f.readlines()
    init_elan=int(init_elan[0])

with open("C:\\Users\\justi\\OneDrive\\Documents\\PAF_2023\\Dataset\\Interactions\\"+n+"\\timestamp-clapperboard.txt") as f:
    init_clap = f.readlines()
    init_clap=int(init_clap[0])


#on lit les annotations du fichier elan
eaf = pympi.Elan.Eaf("C:\\Users\\justi\\OneDrive\\Documents\\PAF_2023\\Dataset\\Interactions\\"+n+"\\"+n+".eaf")
annots0 = sorted(eaf.get_annotation_data_for_tier('Interaction_Form_0'))
annots1 = sorted(eaf.get_annotation_data_for_tier('Interaction_Form_1'))
trust = sorted(eaf.get_annotation_data_for_tier('Trust'))


#on stocke les annotations "Nod" dans nodL
nodL=[]
for annot in annots0:
    if annot[2]=='Nod':
        nodL.append(annot)

for annot in annots1:
    if annot[2]=='Nod':
        nodL.append(annot)

#on convertit les temps des annotations en temps format unix
nodL=[[elem[0]*1000+init_elan,elem[1]*1000+init_elan,elem[2]] for elem in nodL]
trust=[[elem[0]*1000+init_elan,elem[1]*1000+init_elan,elem[2]] for elem in trust]

#on lit le fichier csv
l=[]
csv_file=open("C:\\Users\\justi\\OneDrive\\Documents\\PAF_2023\\Dataset\\Interactions\\"+n+"\\persons.csv","r")
csvreader = csv.reader(csv_file)
i=0
for row in csvreader:
    if row!=[] and i>4:
        if i==5:
            print(row)
        #à partir du numéro de la frame, on passe en temps unix
        #et on prend les données RX,RY,RZ pour les deux personnes
        #attention, ces données correspondent à row[8:11]+row[14:17] pour l'interaction 9
        #mais à row[2:5]+row[8:11] pour les autres interactions (",,,,,," dans le fichier du 9, contre "," pour les autres)
        col=[init_clap+10000*(i-4)]+row[2:5]+row[8:11]
        col=[float(elem) for elem in col]
        if i==5:
            print(col)
        l.append(col)
    i+=1

csv_file.close()



# on lit les données de annotations_person1.xml et on les stocke dans data
with open('C:\\Users\\justi\\OneDrive\\Documents\\PAF_2023\\Dataset\\Interactions\\'+n+'\\annotations_person1.xml', 'r') as f:
    data = f.read() 

# on parse
bs_data = BeautifulSoup(data, features="xml") 

# on trouve et stocke les instances de vfoa, avec le temps en millisecondes associé

b_duree=bs_data.find_all('milliseconds')
b_duree = [span.get_text() for span in b_duree]

b_vfoa = bs_data.find_all('vfoa')
b_vfoa = [span.get_text() for span in b_vfoa]

b_data1=[]

#on concatène les deux
for i in range(len(b_duree)):
    if b_vfoa[i]!=b_vfoa[0]:
        b_data1.append([str(b_duree[i]),str(b_vfoa[i])])

#on convertit en temps unix
for i in range(len(b_data1)-1):
    b_data1[i]=[float(b_data1[i][0])*1000+init_clap,float(b_data1[i+1][0])*1000+init_clap,b_data1[i][1]]
b_data1[-1]=[float(b_data1[-1][0])*1000+init_clap,2*init_clap,b_data1[-1][1]]


#on recommence avec annotations_person2.xml
with open('C:\\Users\\justi\\OneDrive\\Documents\\PAF_2023\\Dataset\\Interactions\\'+n+'\\annotations_person2.xml', 'r') as f:
    data = f.read() 

bs_data = BeautifulSoup(data, features="xml") 

# on trouve et stocke les instances de vfoa, avec le temps en millisecondes associé
b_duree=bs_data.find_all('milliseconds')
b_duree = [span.get_text() for span in b_duree]

b_vfoa = bs_data.find_all('vfoa')
b_vfoa = [span.get_text() for span in b_vfoa]

b_data2=[]

#on concatène les deux
for i in range(len(b_duree)):
    if b_vfoa[i]!=b_vfoa[0]:
        b_data2.append([str(b_duree[i]),str(b_vfoa[i])])

#on convertit en temps unix
for i in range(len(b_data2)-1):
    b_data2[i]=[float(b_data2[i][0])*1000+init_clap,float(b_data2[i+1][0])*1000+init_clap,b_data2[i][1]]
b_data2[-1]=[float(b_data2[-1][0])*1000+init_clap,2*init_clap,b_data2[-1][1]]

b_data=b_data1+b_data2




#calcule à partir des données le vecteur des features sur l'intervalle de temps [debut, debut+duree]
def vecteur(debut, duree,nodL,l, b_data):

    #on conserve les regards qui sont dans la période de temps examinée
    gaze=[]
    for elem in b_data:
        if (elem[0]>=debut and elem[0]<=(debut +duree)):
            if len(gaze)>0 and gaze[-1][2]==elem[2]:
                gaze[-1][1]=elem[1]
            else:
                gaze.append(elem)

    #on conserve les hochements de tête qui sont dans la période de temps examinée
    nod=[]
    for elem in nodL:
        if (elem[0]>=debut and elem[0]<=(debut + duree)):
            if elem[1]>(debut +duree):
                nod.append([elem[0],debut+duree,elem[2]])
            else:
                nod.append(elem)

    #on conserve les angles de position qui sont dans la période de temps examinée    
    head=[]
    for elem in l:
        if (elem[0]>=debut and elem[0]<=(debut + duree)):
            head.append(elem)
    
    #on calcule la vitesse de giration de la tête en "dérivant" l'angle
    vit=[]
    for i in range(1,len(head)):
        vit.append([head[i][0],head[i][1]-head[i-1][1],head[i][2]-head[i-1][2],head[i][3]-head[i-1][3],
                    head[i][4]-head[i-1][4],head[i][5]-head[i-1][5],head[i][6]-head[i-1][6]])

    #vecgaze est le vecteur des regards, une donnée catégorique : vaut 1 si le type de regard est présent
    vecgaze=[0,0,0,0,0,0,0,0,0] # [Dontknow,NotVisible,Object1,Object2,Object3,Person1,Person2,Robot,Unfocused]
    for elem in gaze:
        if elem[2]=='Dontknow':
            vecgaze[0]=1
        if elem[2]=='NotVisible':
            vecgaze[1]=1
        if elem[2]=='Object1':
            vecgaze[2]=1
        if elem[2]=='Object2':
            vecgaze[3]=1
        if elem[2]=='Object3':
            vecgaze[4]=1
        if elem[2]=='Person1':
            vecgaze[5]=1
        if elem[2]=='Person2':
            vecgaze[6]=1
        if elem[2]=='Robot':
            vecgaze[7]=1
        if elem[2]=='Unfocused':
            vecgaze[8]=1
    
    #s'il y a eu un hochement de tête, on met vecnod à 1
    if nod!=[]:
        vecnod=[1]
    else:
        vecnod=[0]
    
    #on récupère le vecteur de la moyenne et de la variance pour les angles et la vitesse
    vechead=meanecart(head)
    vecvit=meanecart(vit)

    #on renvoie la concaténation des vecteurs
    return(vecnod+vecgaze+vechead+vecvit)



#renvoie le vecteur contenant la moyenne et la variance en x,y et z
def meanecart(head):
    
    #les lignes suivantes calculaient le max et le min en x, y et z pour les deux personnes
    #elles ont été enlevées pour raccourcir le vecteur
    #meanhead=[0,0,0,0,0,0]
    #maxxyz=[max([sublist[1] for sublist in head]),max([sublist[2] for sublist in head]),max([sublist[3] for sublist in head]),
    #        max([sublist[4] for sublist in head]),max([sublist[5] for sublist in head]),max([sublist[6] for sublist in head])]
    #minxyz=[min([sublist[1] for sublist in head]),min([sublist[2] for sublist in head]),min([sublist[3] for sublist in head]),
    #        min([sublist[4] for sublist in head]),min([sublist[5] for sublist in head]),min([sublist[6] for sublist in head])]

    #on calcule la moyenne en x,y et z pour les deux acteurs en faisant la somme puis en divisant par la length
    for elem in head:
        meanhead[0]+=elem[1]
        meanhead[1]+=elem[2]
        meanhead[2]+=elem[3]
        meanhead[3]+=elem[4]
        meanhead[4]+=elem[5]
        meanhead[5]+=elem[6]
    meanhead=[elem/len(head) for elem in meanhead]

    #on calcule la variance
    varhead=[0,0,0,0,0,0]
    for elem in head:
        varhead[0]+=(elem[1]-meanhead[0])**2
        varhead[1]+=(elem[2]-meanhead[1])**2
        varhead[2]+=(elem[3]-meanhead[2])**2
        varhead[3]+=(elem[4]-meanhead[3])**2
        varhead[4]+=(elem[5]-meanhead[4])**2
        varhead[5]+=(elem[6]-meanhead[5])**2
    varhead=[elem/len(head) for elem in varhead]

    #on renvoie le vecteur des moyennes et variances
    return(meanhead+varhead)


#on calcule le vecteur pour chaque segment et on le save dans le bon dossier
for i in range(len(trust)):

    x=vecteur(trust[i][0],trust[i][1]-trust[i][0],nodL,l,b_data)
    
    np.save("C:\\Users\\justi\\Data_geste_2\\"+n+"\\segment_"+str(i),x)
