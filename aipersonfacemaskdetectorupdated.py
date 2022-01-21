import os
import torch
import torchvision
import pandas as pd
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools
import seaborn as sns
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

# location of datasets
data_dir = '../input/finaldataset/finalDataset/data/'
biasSet_dir = '../input/finaldataset/finalDataset/biasSet/biasData'
newDataSet_dir = '../input/finaldataset/finalDataset/newData'
classes = os.listdir(data_dir + "train")


## Create transformations for the images in dataset to make them similar in dimensions and normalize
transformations = Compose([
    Resize(255),
    CenterCrop(224),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Load Train and test dataset using ImageFolder from torchvision
dataset = ImageFolder(data_dir+'/train', transform=transformations)
testset = ImageFolder(data_dir+'/test',transform =  transformations)
# Updated data set
newdataset = ImageFolder(newDataSet_dir+'/train', transform=transformations)
newtestset = ImageFolder(newDataSet_dir+'/test',transform =  transformations)
# Split data according to gender
maleTestset = ImageFolder(biasSet_dir+'/gender/male/',transform =  transformations)
femaleTestset = ImageFolder(biasSet_dir+'/gender/female/',transform =  transformations)

class Classification(nn.Module):
    def trainStep(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validationStep(self, batch):
        images, labels = batch 
        out = self(images)   
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validationPerEpoch(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean() 
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class MaskClassifierModel(Classification):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 112 x 112

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 56 x 56

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 28 x 28
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(256 * 28 * 28, 1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 3)
        )
        
    def forward(self, xb):
        xb = self.conv_layer(xb)
        xb = xb.view(xb.size(0), -1)
        xb = self.fc_layer(xb)
        return xb

# Helper methods to train on GPU or CPU
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validationStep(batch) for batch in val_loader]
    return model.validationPerEpoch(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    bestModel = model
    best_accuracy = -1
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.trainStep(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

        #Save the best model         
        if result['val_acc']>best_accuracy:
          print('Last best accuracy: '+ str(best_accuracy)+' New best accuracy: '+str(result['val_acc'])+' Epoch: '+str(epoch))
          bestModel = copy.deepcopy(model)
          best_accuracy = result['val_acc']
    
    
    return bestModel

# Predict class of single image
def predictClass(img, model):
    device = get_default_device()
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return dataset.classes[preds[0].item()]


# predict class for full test set
def predict_class_full(model, test_dl_full):
  prediction= []
  true_labels = []
  for batch in test_dl_full:
    images , labels = batch
    yb = model(images)
    _, preds  = torch.max(yb, dim=1)
    prediction.append(preds.item())
    #true_labels.append(labels)
  return prediction


def trainModel(model,dataSetUsed,idx):
  #Define device to be used
  device = get_default_device()

  #Load data
  random_seed = 42
  torch.manual_seed(random_seed)
  val_size = int(len(dataSetUsed)*0.10)
  train_size = len(dataSetUsed) - val_size
  train_ds, val_ds = random_split(dataSetUsed, [train_size, val_size])
  

  # Setting Hyperparameters
  num_epochs = 10
  opt_func = torch.optim.Adam
  lr = 0.001
  batch_size=32

  train_dl = DataLoader(train_ds, batch_size, shuffle=True)
  val_dl = DataLoader(val_ds, batch_size*2)
  train_dl = DeviceDataLoader(train_dl, device)
  val_dl = DeviceDataLoader(val_dl, device)
  to_device(model, device);

  bestModel = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
  torch.save(bestModel.state_dict(), 'FaceMaskClassifier'+idx+'.pth')
  print("Model Architecture:")
  print(bestModel)

  # Create Image for CNN model architecture
  #dummBatch = next(iter(train_dl))
  #images, labels = dummBatch
  #yDum = model(images)
  #make_dot(yDum, params=dict(list(model.named_parameters()))).render("cnn_torchviz", format="png")

  # Show prediction on one image in test set
  img, label = testset[504]
  plt.imshow(img.permute(1, 2, 0))
  print('Label:', dataset.classes[label], ', Predicted:', predictClass(img, bestModel))
  return bestModel

def testModel(model,testSetUsed):
  #Define device to be used
  device = get_default_device()
  calculatedMetrics = {}
  test_dl_full = DataLoader(testSetUsed)
  test_dl_full = DeviceDataLoader(test_dl_full, device)
  to_device(model, device);
  prediction= predict_class_full(model, test_dl_full)
  ytest = np.array([y for x, y in iter(testSetUsed)])
  calculatedMetrics["Accuracy"] = accuracy_score(ytest, prediction)
  prediction_tensor = torch.from_numpy(np.array(prediction))
  label_tensor = torch.from_numpy(ytest)

  # Create confusion matrix
  cm = confusion_matrix(label_tensor, prediction_tensor, normalize= 'true')
  calculatedMetrics["ConfusionMatrix"] = cm
  calculatedMetrics["F1"] = f1_score(ytest, prediction,average = 'weighted')
  calculatedMetrics["Recall"] = recall_score(ytest,prediction, average = 'weighted')
  calculatedMetrics["Precision"] = precision_score(ytest,prediction, average='weighted')
  print("----------------------------PERFORMANCE ON TEST REPORT BEGIN----------------------------------")
  print(calculatedMetrics)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=testset.classes)
  disp.plot()  
  print("----------------------------PERFORMANCE ON TEST REPORT END----------------------------------")
  return calculatedMetrics

def loadModelAndTest(testsetUset):
  # Loading saved model and evaluating performance
  device = get_default_device()
  model2 = to_device(MaskClassifierModel(), device)
  model2.load_state_dict(torch.load('')) #path of model on file system
  testModel(model2,testsetUset)
  img, label = testsetUset[504]
  plt.imshow(img.permute(1, 2, 0))
  print('Label:', dataset.classes[label], ', Predicted:', predictClass(img, model2))



def trainModelKFold(model,dataSetUsed):
  #Define device to be used
  device = get_default_device()

  #Load data
  random_seed = 42
  torch.manual_seed(random_seed)
  val_size = int(len(dataSetUsed)*0.10)
  train_size = len(dataSetUsed) - val_size
  train_ds, val_ds = random_split(dataSetUsed, [train_size, val_size])
  

  # Setting Hyperparameters
  num_epochs = 10
  opt_func = torch.optim.Adam
  lr = 0.001
  batch_size=32

  train_dl = DataLoader(train_ds, batch_size, shuffle=True)
  val_dl = DataLoader(val_ds, batch_size*2)
  train_dl = DeviceDataLoader(train_dl, device)
  val_dl = DeviceDataLoader(val_dl, device)
  to_device(model, device);

  bestModel = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
  return bestModel


# Combine train and test set
trainAndTestSet = []
trainAndTestSet.append(dataset)
trainAndTestSet.append(testset)
trainAndTestSet = torch.utils.data.ConcatDataset(trainAndTestSet)

# Combine updated train and test set
newTrainAndTestSet = []
newTrainAndTestSet.append(newdataset)
newTrainAndTestSet.append(newtestset)
newTrainAndTestSet = torch.utils.data.ConcatDataset(newTrainAndTestSet)


def calculateAveragePerformance(AllFoldMetrics):
    res = {}
    size = len(AllFoldMetrics)
    total_acc = 0
    total_precision = 0
    total_recall = 0
    total_f1=0
    list_of_confusionMatrix = []
    for metric in AllFoldMetrics:
        total_acc = total_acc + metric['Accuracy']
        total_f1 = total_f1+metric['F1']
        total_recall = total_recall+metric['Recall']
        total_precision =total_precision + metric['Precision']
        list_of_confusionMatrix.append(metric['ConfusionMatrix'])
    
    mean_conf_matrix = np.mean(list_of_confusionMatrix, axis=0)
    disp = ConfusionMatrixDisplay(confusion_matrix=mean_conf_matrix,display_labels=['notPerson','withMask','withoutMask'])
    disp.plot()
    res['ConfusionMatrix'] = mean_conf_matrix
    res['Accuracy'] = total_acc/size
    res['Recall'] = total_recall/size
    res['Precision'] = total_precision/size
    res['F1'] = total_f1/size
    
    return res

def PerformKFoldEvaluation(sampleData,idx):
  kf = KFold(n_splits=10,shuffle=True,random_state=None)
  bestModel = None
  AllFoldMetrics = []
  bestAccuracy = -1
  for fold, (train_index, test_index) in enumerate(kf.split(sampleData)):
    print('--------- Fold '+str(fold)+' ---------')
    trainedModel = trainModelKFold(model= MaskClassifierModel(),dataSetUsed= torch.utils.data.dataset.Subset(sampleData,train_index))
    calculatedMetrics = testModel(trainedModel,torch.utils.data.dataset.Subset(sampleData,test_index))
    if(calculatedMetrics["Accuracy"]>bestAccuracy):
      bestAccuracy = calculatedMetrics["Accuracy"]
      bestModel = copy.deepcopy(trainedModel)
    
    AllFoldMetrics.append(calculatedMetrics)
  
  torch.save(bestModel.state_dict(), 'FaceMaskClassifierKFold'+idx+'.pth')
  print("--------------------AGGREGATE PERFORMANCE----------------------------")  
  print(calculateAveragePerformance(AllFoldMetrics))

#Train model on original data
model = MaskClassifierModel()
model = trainModel(model,dataset,'First')

initResult=testModel(model,testset)


# Detect bias
resMale = testModel(model,maleTestset)
resFemale = testModel(model,femaleTestset)

# Retraining model on new data
modelNew = MaskClassifierModel()
modelNew = trainModel(modelNew,newdataset,'Second')

resultNewData = testModel(modelNew,newtestset)


# Recheck Bias
resultMale = testModel(modelNew,maleTestset)
resultFemale = testModel(modelNew,femaleTestset)

PerformKFoldEvaluation(trainAndTestSet,'First')
PerformKFoldEvaluation(newTrainAndTestSet,'Second')

