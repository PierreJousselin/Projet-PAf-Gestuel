import pympi
import numpy as np
from Pierre_bras_dans_le_dos import get_data_dos
from Pierre_bras_croisé_visage import get_data_front

#Sauveagrde les données gesteulle des bras sous format npy

for i in [9,10,12,15,18,19,24,26,27,30]:
    # file: dossier de l'interrecation dans lequelle on se trouve
    # annots liste des annotation 
    file ="C://Users//Pierre//Documents//PAF//PAF_2023//Dataset//Interactions//"+str(i)+"//"+str(i)+".eaf" 
    eaf = pympi.Elan.Eaf(file)
    annots = sorted(eaf.get_annotation_data_for_tier('Trust')) 
    print(annots)
    
    num_segment=0
    for segment in annots:
    
        saved_file="C://Users//Pierre//Documents//PAF//Data_geste//"+str(i)+"//segment_"+str(num_segment)
        data_dos=get_data_dos(segment,i)
        data_croisé,data_face=get_data_front(segment,i)
        data=[data_dos,data_croisé,data_face]

        data=np.save(saved_file,np.array(data))

        num_segment+=1
        