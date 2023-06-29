import subprocess
import os

# Chemin vers le fichier exécutable d'OpenPose
openpose_executable = r"C://Users//Pierre//Documents//PAF//openpose//bin//OpenPoseDemo.exe"
os.chdir("C://Users//Pierre//Documents//PAF//openpose")

# Arguments de la commande
arguments = [
    "--video", "C://Users//Pierre//Documents//PAF//PAF_2023//Dataset//Interactions//30//cam-table-wall.mp4",
    "--write_json", "C://Users//Pierre//Documents//PAF//openpose//output30wall",
    "--render_pose", "0",
    "--display", "0"
]

# Construction de la commande complète
command = [openpose_executable] + arguments

# Exécution de la commande pour une frame sur deux
subprocess.run(command + ["--frame_step", "10"], shell=True)
