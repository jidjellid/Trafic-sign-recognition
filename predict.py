import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import natsort
from functools import reduce 
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from skimage.transform import resize
import numpy as np

classes = { 
    0:"Limitation de vitesse (20km/h)",
    1:"Limitation de vitesse (30km/h)", 
    2:"Limitation de vitesse (50km/h)", 
    3:"Limitation de vitesse (60km/h)", 
    4:"Limitation de vitesse (70km/h)", 
    5:"Limitation de vitesse (80km/h)", 
    6:"Fin de limitation de vitesse (80km/h)", 
    7:"Limitation de vitesse (100km/h)", 
    8:"Limitation de vitesse (120km/h)", 
    9:"Interdiction de depasser", 
    10:"Interdiction de depasser pour vehicules > 3.5t", 
    11:"Intersection ou' vous etes prioritaire", 
    12:"Route prioritaire", 
    13:"Ceder le passage", 
    14:"Arret a' l'intersection", 
    15:"Circulation interdite", 
    16:"Acces interdit aux vehicules > 3.5t", 
    17:"Sens interdit", 
    18:"Danger", 
    19:"Virage a' gauche", 
    20:"Virage a' droite", 
    21:"Succession de virages", 
    22:"Cassis ou dos-d'ane", 
    23:"Chaussee glissante", 
    24:"Chaussee retrecie par la droite", 
    25:"Travaux en cours", 
    26:"Annonce feux", 
    27:"Passage pietons", 
    28:"Endroit frequente' par les enfants", 
    29:"Debouche' de cyclistes", 
    30:"Neige ou glace",
    31:"Passage d'animaux sauvages", 
    32:"Fin des interdictions precedemment signalees", 
    33:"Direction obligatoire a' la prochaine intersection : a' droite", 
    34:"Direction obligatoire a' la prochaine intersection : a' gauche", 
    35:"Direction obligatoire a' la prochaine intersection : tout droit", 
    36:"Direction obligatoire a' la prochaine intersection : tout droit ou a' droite", 
    37:"Direction obligatoire a' la prochaine intersection : tout droit ou a' gauche", 
    38:"Contournement obligatoire de l'obstacle par la droite", 
    39:"Contournement obligatoire de l'obstacle par la gauche", 
    40:"Carrefour giratoire", 
    41:"Fin d'interdiction de depasser", 
    42:"Fin d'interdiction de depasser pour vehicules > 3.5t" 
}

trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor()
    ])

#Dropout inutile, mais pas envie de changer le modele pour ca
class LeNet5(nn.Module):
    def __init__(self,dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3))    
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3))
        self.dropout1 = nn.Dropout(p=dropout_p) # as attribute, to be activated/disactivated by model.eval()/train()
        self.batchnorm1 = nn.BatchNorm2d(32)    # as attribute, for affine=True
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))    
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3)) 
        self.dropout2 = nn.Dropout(p=dropout_p) # as attribute, to be activated/disactivated by model.eval()/train()
        self.batchnorm2 = nn.BatchNorm2d(64)    # as attribute, for affine=True
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3))    
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3)) 
        self.dropout3 = nn.Dropout(p=dropout_p) # as attribute, to be activated/disactivated by model.eval()/train()
        self.batchnorm3 = nn.BatchNorm2d(128)   # as attribute, for affine=True
        self.fc1 = nn.Linear(128, 256)
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 43)

    def forward(self, input):
        layer1 = F.relu(self.conv1(input))                  
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.max_pool2d(layer2, kernel_size=2, stride=2)
        layer3_d = self.dropout1(layer3)
        layer3_db = self.batchnorm1(layer3_d)
        layer4 = F.relu(self.conv3(layer3_db))                  
        layer5 = F.relu(self.conv4(layer4))            
        layer6 = F.max_pool2d(layer5, kernel_size=2, stride=2)
        layer6_d = self.dropout2(layer6)
        layer6_db = self.batchnorm2(layer6_d)
        layer7 = F.relu(self.conv5(layer6_db))                  
        layer8 = F.relu(self.conv6(layer7))
        layer8_d = self.dropout3(layer8)     
        layer8_db = self.batchnorm3(layer8_d)       
        layer9 = F.relu(self.fc1(torch.flatten(layer8_db,1)))  
        layer9_b = self.batchnorm4(layer9)
        layer10 = self.fc2(layer9_b)                        
        return layer10

if not os.path.exists("modelWeights"):
    print("Pas de poids pour le modele trouvés")
    exit(0)

mainModel = LeNet5(dropout_p=0.0)
mainModel.load_state_dict(torch.load("modelWeights"))

#Renvoie la prédiction de model sur un tenseur contenant l'image à prédire
def predict(imageTensor, model):
    model.eval() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    imageTensor = imageTensor.to(device)
    return model(imageTensor.view(-1,3,32,32)).argmax(1)

def predictFolder(path):
    
    class basicFolderDataset(Dataset):
        def __init__(self, main_dir, transform):
            self.main_dir = main_dir
            self.transform = transform
            self.list = []
            for file in os.listdir(main_dir):
                if file.endswith(".png"):
                    self.list.append(file)
            self.list = natsort.natsorted(self.list)

        def __len__(self):
            return len(self.list)

        def __getitem__(self, idx):
            img_loc = os.path.join(self.main_dir, self.list[idx])
            image = Image.open(img_loc).convert("RGB")
            tensor_image = self.transform(image)
            return tensor_image
            
    load = DataLoader(basicFolderDataset(path, transform=trans), batch_size=1024)
    tens = None
    for batch in load:
        if tens is None:
            tens = predict(batch,mainModel)
        else:
            tens = torch.cat((tens,predict(batch,mainModel)),0)
    return tens

def predictImage(path):
    return predict(trans(Image.open(path).convert("RGB")),mainModel).item()

def resultAccuracy(resultPath, labelPath):

    #Read result and compare good answers
    results = pd.read_csv(resultPath)['class']
    labels = pd.read_csv(labelPath)['ClassId']#"panneaux_route/Test.csv"

    correct = 0
    total = 0

    for i in range(len(results)):
        if(results[i] == labels[i]):
            correct += 1
        total += 1

    print("Correct =",correct,"| Total =",total,"|",(correct/total)*100,"% accuracy")

if len(sys.argv) < 2:
    print("Arguments invalides")
    exit(0)
else:

    path = sys.argv[1]
    mainModel.eval()

    if os.path.exists(path):#Check if the path exists
        if os.path.isdir(path):#If the path is a directory  

            #Write predictions in Result.csv
            f = open("Results.csv","w")
            f.write("class, translated\n")
            for res in predictFolder(path):
                f.write(str(res.item())+","+classes.get(res.item())+"\n")
            f.close()

            if os.path.exists(path+".csv"):
                resultAccuracy("Results.csv", path+".csv")
        else:
            print("Prediction : "+classes.get(predictImage(path)))
    else:
        print("Pas de dossier/fichier trouvé au chemin",path)
        exit(0)