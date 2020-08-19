# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 04:43:21 2019

@author: acham
"""

import pandas as pd
import os
from flask import Flask, request, redirect, flash, render_template, send_from_directory, url_for,session
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
from flask import Flask, jsonify, request, render_template
import time
import json
from flask import jsonify
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import PIL
from PIL import Image
from collections import OrderedDict
from werkzeug.utils import secure_filename
import secrets

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from flask_cors import CORS
from MulticlassClassification import MulticlassClassification

secret = secrets.token_urlsafe(32)

app = Flask(__name__)
UPLOAD_FOLDER = "./"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = secret

cors = CORS(app)
df = pd.read_csv("./coursera.csv")
df = df [["Topic", "career_level", "hours", "math_experience", "Category"]]


X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

# Split into train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)

#X_test = X_test[:10]
#y_test = y_test[:10]


# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

def get_class_distribution(obj):
    count_dict = {
        "rating_1": 0,
        "rating_2": 0,
        "rating_3": 0,
    }
    
    for i in obj:
        if i == 1: 
            count_dict['rating_1'] += 1
        elif i == 2: 
            count_dict['rating_2'] += 1
        elif i == 3: 
            count_dict['rating_3'] += 1             
        else:
            print("Check classes.")
            
    return count_dict

class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

target_list = []
for _, t in train_dataset:
    target_list.append(t)
    
target_list = torch.tensor(target_list)
target_list = target_list[torch.randperm(len(target_list))]

class_count = [i for i in get_class_distribution(y_train).values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float)

class_weights = torch.tensor([0.1000, 0.0149, 0.0625, 0.0022])#, 0.0070, 0.0714])


class_weights_all = class_weights[target_list]

weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

EPOCHS = 500
BATCH_SIZE = 18
LEARNING_RATE = 0.0007
NUM_FEATURES = len(X.columns)
NUM_CLASSES = 4

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          sampler=weighted_sampler
)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x
        
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc) * 100
    
    return acc
    
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}


for e in range(1, EPOCHS+1):
    
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = model(X_train_batch)
        
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        
        
    # VALIDATION    
    with torch.no_grad():
        
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))


      
with open('./coursera.json') as f:
    rec = json.load(f)


        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

3# Split into train+val and test
#X = df.iloc[:, 0:-1]
#y = df.iloc[:, -1]

#X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)




class ClassifierDataset(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)







asd = None

import pandas as pd

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/test", methods=["GET", 'POST'])
def about():
    if request.method == 'POST':
        asd = request.json
        print(asd)
        session['newArray'] = asd
        if 'newArray' in session:
            return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
    
    return render_template("test.html")
    

@app.route("/more", methods=["GET"])
def learn():
    return render_template("learn.html")



import json
@app.route('/createform', methods=['GET', 'POST'])
def createform():
    if request.method == 'POST':
        global asd
        asd = request.json
        print(asd)
        session['newArray'] = asd
        if 'newArray' in session:
            return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
    else:
        asd = [1,2,1,1]
    return (asd)

@app.route('/portfolio', methods=['GET'])
def portfolio():
    result = asd
    data = [[1,1,4,3], [1,1,4,3],[1,3,1,1],[1,3,2,2], asd]
    # # Create the pandas DataFrame 
    df = pd.DataFrame(data, columns = ['Topic', 'career_level','hours', 'math_experience'])

    X_test = df

    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)
    X_test = np.array(X_test)
    test_dataset = ClassifierDataset(torch.from_numpy(np.array(X_test)).float())
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    #y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    # result = y_pred_list[-1]
    
    final = []
    rec1 = ''
    for key, value in rec.items():
    # #print(value)
        for k,v in value.items():
            if k == "Category" and v == str(y_pred_list[-1]):
                final.append(value)
            
        rec1 = json.dumps(final)
        
    rec1 = json.loads(rec1)
    return render_template("portfolio.html", result=rec1)
if __name__ == "__main__":
    from MulticlassClassification import MulticlassClassification
    app.run()
