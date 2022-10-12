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

class LeNet5(nn.Module):
    def __init__(self,dropout_p=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3))    
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3))
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.batchnorm1 = nn.BatchNorm2d(32) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))    
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3)) 
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.batchnorm2 = nn.BatchNorm2d(64) 
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3))    
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3)) 
        self.dropout3 = nn.Dropout(p=dropout_p)
        self.batchnorm3 = nn.BatchNorm2d(128)
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

mainModel = LeNet5(dropout_p=0.5)

def train_loop(train_loader, validation_loader, model, loss_map, lr=1e-3, epochs=20, weight_decay=0.0):
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history=[]

    bestModel = mainModel.state_dict()
    bestValidationAcc = 0
    # Train model
    model.train() 
    for epoch in range(epochs):
        loss_epoch = 0.
        previousTime = time.time()
        for images, labels in train_loader:
            # Transfers data to GPU
            images, labels = images.to(device), labels.to(device)
            # Primal computation
            output = model(images)            
            loss = loss_map(output, labels)            
            # Gradient computation
            model.zero_grad()
            loss.backward()
            # perform parameter update based on current gradients
            optimizer.step()
            # compute the epoch training loss
            loss_epoch += loss.item()
        # display the epoch training loss
        currentTime = time.time()

        validationAccuracy = round(validate(validation_loader, model).item(),3)
        trainingAccuracy = round(validate(train_loader, model).item(),3)

        history.append({"epoch" : epoch, "loss" : loss_epoch, "train_acc" : trainingAccuracy, "val_acc" : validationAccuracy})
        print(f"epoch : {epoch + 1}/{epochs}, loss = {loss_epoch:.6f}, validation_accuracy = {validationAccuracy}% , train_accuracy = {trainingAccuracy}%, epoch_time = {(currentTime - previousTime):.1f}s, time_remaining = {((currentTime - previousTime)*(epochs - epoch)):.1f}s")
        if(validationAccuracy > bestValidationAcc):
            bestModel = mainModel.state_dict()
            bestValidationAcc = validationAccuracy
            print(f"New best model with {validationAccuracy}% accuracy on validation set !")
    
    mainModel.load_state_dict(bestModel)

#Renvoie le taux de réussite d'un model sur un dataloader contenant un tuple (images,labels)         
def validate(data_loader, model):
    nb_errors = 0
    nb_tests = 0
    device = next(model.parameters()).device # current model device
    model.eval()
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device) # move data same model device
        output = model(images)
        nb_errors += ((output.argmax(1)) != labels).sum()
        nb_tests += len(images)
    
    return torch.div((100*(nb_tests-nb_errors)),nb_tests)

if len(sys.argv) < 2:
    print("Arguments invalides")
    exit(0)
else:

    path = sys.argv[1]

    #Path to Dataset
    trainData = torchvision.datasets.ImageFolder(path, transform=trans)

    #Restore values of labels
    trainData.target_transform = lambda id: int(trainData.classes[id])

    #Division du dataset en train et validation
    trainSize = int(0.8 * len(trainData))
    validationSize = len(trainData) - trainSize
    trainDataset, validationDataset = random_split(trainData, [trainSize, validationSize])
    print("Train dataset length size :",len(trainData))
    print("Train dataset length size :",len(validationDataset))

    #Chargement dans un dataloader
    trainLoader = DataLoader(trainDataset, batch_size=256, shuffle=True)
    validationLoader = DataLoader(validationDataset, batch_size=1024)

    train_loop(
        train_loader=trainLoader,
        validation_loader=validationLoader,
        model=mainModel, 
        loss_map=nn.CrossEntropyLoss(),
        lr=0.0015, #augmentation du lr pour une descente de grad plus rapide 
        epochs=10,
        weight_decay=0.000125)
    print("Saving model")

    torch.save(mainModel.state_dict(), "modelWeights")
