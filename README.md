### GTSRB
A CNN (Convolutional Neural Network) for German traffic sign image classification.

![First_pic](https://github.com/HoseinNasiriShahraki/GTSRB/blob/main/Examples/00006.png?raw=true "First_pic") ![second_pic](https://github.com/HoseinNasiriShahraki/GTSRB/blob/main/Examples/00009.png?raw=true "First_pic") ![second_pic](https://github.com/HoseinNasiriShahraki/GTSRB/blob/main/Examples/00042.png?raw=true "First_pic") ![second_pic](https://github.com/HoseinNasiriShahraki/GTSRB/blob/main/Examples/00051.png?raw=true "First_pic") ![second_pic](https://github.com/HoseinNasiriShahraki/GTSRB/blob/main/Examples/00086.png?raw=true "First_pic") ![second_pic](https://github.com/HoseinNasiriShahraki/GTSRB/blob/main/Examples/00093.png?raw=true "First_pic") ![second_pic](https://github.com/HoseinNasiriShahraki/GTSRB/blob/main/Examples/00174.png?raw=true "First_pic")


GTSRB (German Traffic Sign Recognition) is a german traffic sign dataset with 43 classes.
The DataSet has 51,839 pictures of Different Traffic signs with various Brightness levels, image sizes, camera angles......


We begin our process with importing our needed libraries.

```python
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
```
## DataSet
```python
data_dir = "./GTSRB"
train_path = "./Train/"
test_path = "./Test/"
meta_path = "./Meta/"

meta_df = pd.read_csv('./GTSRB/Meta.csv')
train_df = pd.read_csv('./GTSRB/Train.csv')
test_df = pd.read_csv('./GTSRB/Test.csv')
```


We transform our images using Torch Transform library.
With Transforms.Compose we can implement various Transforms and Augmentation to our images.
```pyton
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize((28,28)),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
```

We Create our DataSet Class and define the "__len__" and "__get__item__" functions.
```python
#creating the dataset class
class GTSR_DataSet(Dataset):
    def __init__(self, df, root_dir,transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        image_path = os.path.join(self.root_dir,self.df.iloc[index,7])  #the column of paths in dataframe is 7
        image = Image.open(image_path)
        y_class = torch.tensor(self.df.iloc[index, 6]) #the column of ClsassId in daraframe is 6
        
        if self.transform:
            image = self.transform(image)
            return (image, y_class)
```
Creating our Train and Test DataSet Object.
And we use DataLoaders to Properly Load and implement our Data into the Model Later.
```python
training_set = GTSR_DataSet(train_df,data_dir,transform=transforms)
test_set = GTSR_DataSet(test_df,data_dir,transform=transforms)
```
```python
train_loader = DataLoader(dataset = training_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_set, batch_size=batch_size, shuffle=False)

dataloaders = {'training':train_loader,'testing':test_loader}
dataset_sizes = {'training':len(train_loader.dataset),'testing':len(test_loader.dataset)}
print(dataset_sizes)
```
output:
```
{'training': 39209, 'testing': 12630}
```
## Model
We define every Layer of our DeepLearning Procces in our Model Class:
Then We Create our Desiered image Proccesing Pipeline in "__Forward__" Method. 
```python
#creating the model
class GTRSB_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(GTRSB_Model, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.batchnorm2 = nn.BatchNorm2d(256)

        self.l1 = nn.Linear(12544,512)
        self.l2 = nn.Linear(512,128)
        self.batchnorm4 = nn.LayerNorm(128)
        self.l3 = nn.Linear(128,output_size)




    def forward(self,input):
        #training pipeline

        conv = self.conv1(input)
        conv = self.conv2(conv)

        batchnorm = self.relu(self.batchnorm1(conv))
        maxpool = self.maxpool(batchnorm)

        conv = self.conv3(maxpool)
        conv = self.conv4(conv)

        batchnorm = self.relu(self.batchnorm2(conv))
        maxpool = self.maxpool(batchnorm)
        flatten = self.flatten(maxpool)
        
        #Neural Network Featuremap input
        dense_l1 = self.l1(flatten)
        dropout = self.dropout3(dense_l1)
        dense_l2 = self.l2(dropout)
        batchnorm = self.batchnorm4(dense_l2)
        dropout = self.dropout2(batchnorm)
        output = self.l3(dropout)
        
    
        return output
```
Defining our Hyper Parameters And Creating Our Model And Setting Our Device To Operate On GPU Using Nvidia's CUDA:
```python
input_size = 3*28*28
output_size = 43
num_epochs = 2
batch_size = 64
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GTRSB_Model(input_size=input_size, output_size=output_size)
model.cuda()
```
output:
```
GTRSB_Model(
  (relu): ReLU()
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (dropout2): Dropout(p=0.2, inplace=False)
  (dropout3): Dropout(p=0.3, inplace=False)
  (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (batchnorm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (batchnorm2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (l1): Linear(in_features=12544, out_features=512, bias=True)
  (l2): Linear(in_features=512, out_features=128, bias=True)
  (batchnorm4): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  (l3): Linear(in_features=128, out_features=43, bias=True)
)
```




Our model is Now Complete and ready to Operate.
But first We need a Criterion And an Optimizer.
There are so many Optimizers in deeplearning like __Adam__ , __SGD__ , __ASGD__ And others....
Therefore I Wasn't Certain Wich One to Choose, So I Used __ALL OF THEM__:

```python
# model deployment with SGD Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate , momentum=0.9)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # images, labels = images.type(torch.LongTensor), labels.type(torch.LongTensor)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 500 == 0:
        #     print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item()}')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples +=labels.shape[0]
            n_correct +=(predictions == labels).sum().item()
        acc = 100.0 * (n_correct / n_samples)
        print(f'epoch {epoch+1} / 10, SGD accuracy = {acc}')
        
        

print("---------------------------------")


        
        

# model deployment with Adam optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # images, labels = images.type(torch.LongTensor), labels.type(torch.LongTensor)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 500 == 0:
        #     print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item()}')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples +=labels.shape[0]
            n_correct +=(predictions == labels).sum().item()
        acc = 100.0 * (n_correct / n_samples)
        print(f'epoch {epoch+1} / 10, ADAM accuracy = {acc}')
        
        
        

print("---------------------------------")


        

# model deployment with ASGD Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # images, labels = images.type(torch.LongTensor), labels.type(torch.LongTensor)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 500 == 0:
        #     print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item()}')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples +=labels.shape[0]
            n_correct +=(predictions == labels).sum().item()
        acc = 100.0 * (n_correct / n_samples)
        print(f'epoch {epoch+1} / 10, ASGD accuracy = {acc}')
        
        
        

print("---------------------------------")




# model deployment with Adadelta Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate , rho=0.9)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # images, labels = images.type(torch.LongTensor), labels.type(torch.LongTensor)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 500 == 0:
        #     print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item()}')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples +=labels.shape[0]
            n_correct +=(predictions == labels).sum().item()
        acc = 100.0 * (n_correct / n_samples)
        print(f'epoch {epoch+1} / 10, Adadelta accuracy = {acc}')
        
        

print("---------------------------------")


# model deployment with Adagrad Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # images, labels = images.type(torch.LongTensor), labels.type(torch.LongTensor)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 500 == 0:
        #     print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item()}')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples +=labels.shape[0]
            n_correct +=(predictions == labels).sum().item()
        acc = 100.0 * (n_correct / n_samples)
        print(f'epoch {epoch+1} / 10, Adagrad accuracy = {acc}')
        
        

print("---------------------------------")

        
        
# model deployment with AdamW Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # images, labels = images.type(torch.LongTensor), labels.type(torch.LongTensor)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 500 == 0:
        #     print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item()}')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples +=labels.shape[0]
            n_correct +=(predictions == labels).sum().item()
        acc = 100.0 * (n_correct / n_samples)
        print(f'epoch {epoch+1} / 10, AdamW accuracy = {acc}')
        
        

print("---------------------------------")


        
        
# model deployment with Adamax Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # images, labels = images.type(torch.LongTensor), labels.type(torch.LongTensor)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 500 == 0:
        #     print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item()}')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples +=labels.shape[0]
            n_correct +=(predictions == labels).sum().item()
        acc = 100.0 * (n_correct / n_samples)
        print(f'epoch {epoch+1} / 10, Adamax accuracy = {acc}')
        
        

print("---------------------------------")        

        
# model deployment with NAdam Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # images, labels = images.type(torch.LongTensor), labels.type(torch.LongTensor)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 500 == 0:
        #     print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item()}')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples +=labels.shape[0]
            n_correct +=(predictions == labels).sum().item()
        acc = 100.0 * (n_correct / n_samples)
        print(f'epoch {epoch+1} / 10, NAdam accuracy = {acc}')
        
        

print("---------------------------------")


        
# model deployment with RAdam Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # images, labels = images.type(torch.LongTensor), labels.type(torch.LongTensor)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 500 == 0:
        #     print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item()}')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples +=labels.shape[0]
            n_correct +=(predictions == labels).sum().item()
        acc = 100.0 * (n_correct / n_samples)
        print(f'epoch {epoch+1} / 10, RAdam accuracy = {acc}')
        
        
        
print("---------------------------------")

        
# model deployment with RMSprop Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # images, labels = images.type(torch.LongTensor), labels.type(torch.LongTensor)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 500 == 0:
        #     print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item()}')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples +=labels.shape[0]
            n_correct +=(predictions == labels).sum().item()
        acc = 100.0 * (n_correct / n_samples)
        print(f'epoch {epoch+1} / 10, RMSprop accuracy = {acc}')
        


print("---------------------------------")

        
        
# model deployment with Rprop Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # images, labels = images.type(torch.LongTensor), labels.type(torch.LongTensor)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 500 == 0:
        #     print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item()}')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples +=labels.shape[0]
            n_correct +=(predictions == labels).sum().item()
        acc = 100.0 * (n_correct / n_samples)
        print(f'epoch {epoch+1} / 10, Rprop accuracy = {acc}')
```
After 2 Hours of Trainning Here Are The Final Results:
```
epoch 1 / 10, SGD accuracy = 91.07680126682503
epoch 2 / 10, SGD accuracy = 93.95882818685669
epoch 3 / 10, SGD accuracy = 95.49485352335708
epoch 4 / 10, SGD accuracy = 95.53444180522565
epoch 5 / 10, SGD accuracy = 95.71654790182106
epoch 6 / 10, SGD accuracy = 96.27870150435471
epoch 7 / 10, SGD accuracy = 96.3341250989707
epoch 8 / 10, SGD accuracy = 96.41330166270784
epoch 9 / 10, SGD accuracy = 96.40538400633413
epoch 10 / 10, SGD accuracy = 96.57957244655582
---------------------------------
epoch 1 / 10, ADAM accuracy = 94.48139350752177
epoch 2 / 10, ADAM accuracy = 95.63737133808394
epoch 3 / 10, ADAM accuracy = 95.23357086302454
epoch 4 / 10, ADAM accuracy = 94.98020585906572
epoch 5 / 10, ADAM accuracy = 95.81947743467933
epoch 6 / 10, ADAM accuracy = 95.89865399841648
epoch 7 / 10, ADAM accuracy = 94.64766429136976
epoch 8 / 10, ADAM accuracy = 95.57403008709421
epoch 9 / 10, ADAM accuracy = 97.06254948535233
epoch 10 / 10, ADAM accuracy = 96.04908946951703
---------------------------------
epoch 1 / 10, ASGD accuracy = 96.31037212984957
epoch 2 / 10, ASGD accuracy = 96.326207442597
epoch 3 / 10, ASGD accuracy = 96.57165479018211
epoch 4 / 10, ASGD accuracy = 96.48456057007125
epoch 5 / 10, ASGD accuracy = 96.54790182106096
epoch 6 / 10, ASGD accuracy = 96.55581947743468
epoch 7 / 10, ASGD accuracy = 96.65874901029295
epoch 8 / 10, ASGD accuracy = 96.70625494853523
epoch 9 / 10, ASGD accuracy = 96.78543151227237
epoch 10 / 10, ASGD accuracy = 96.66666666666667
---------------------------------
epoch 1 / 10, Adadelta accuracy = 96.66666666666667
epoch 2 / 10, Adadelta accuracy = 96.69833729216151
epoch 3 / 10, Adadelta accuracy = 96.60332541567695
epoch 4 / 10, Adadelta accuracy = 96.83293745051465
epoch 5 / 10, Adadelta accuracy = 96.71417260490894
epoch 6 / 10, Adadelta accuracy = 96.8012668250198
epoch 7 / 10, Adadelta accuracy = 96.70625494853523
epoch 8 / 10, Adadelta accuracy = 96.75376088677751
epoch 9 / 10, Adadelta accuracy = 96.71417260490894
epoch 10 / 10, Adadelta accuracy = 96.72209026128266
---------------------------------
epoch 1 / 10, Adagrad accuracy = 97.36342042755345
epoch 2 / 10, Adagrad accuracy = 97.37925574030088
epoch 3 / 10, Adagrad accuracy = 97.41884402216944
epoch 4 / 10, Adagrad accuracy = 97.513855898654
epoch 5 / 10, Adagrad accuracy = 97.42676167854314
epoch 6 / 10, Adagrad accuracy = 97.64845605700712
epoch 7 / 10, Adagrad accuracy = 97.59303246239112
epoch 8 / 10, Adagrad accuracy = 97.70387965162311
epoch 9 / 10, Adagrad accuracy = 97.6959619952494
epoch 10 / 10, Adagrad accuracy = 97.70387965162311
---------------------------------
epoch 1 / 10, AdamW accuracy = 97.2763262074426
epoch 2 / 10, AdamW accuracy = 96.0332541567696
epoch 3 / 10, AdamW accuracy = 96.31037212984957
epoch 4 / 10, AdamW accuracy = 96.59540775930324
epoch 5 / 10, AdamW accuracy = 97.31591448931115
epoch 6 / 10, AdamW accuracy = 97.61678543151227
epoch 7 / 10, AdamW accuracy = 95.47901821060965
epoch 8 / 10, AdamW accuracy = 97.26840855106889
epoch 9 / 10, AdamW accuracy = 96.95170229612035
epoch 10 / 10, AdamW accuracy = 96.98337292161521
---------------------------------
epoch 1 / 10, Adamax accuracy = 98.08392715756136
epoch 2 / 10, Adamax accuracy = 98.06017418844021
epoch 3 / 10, Adamax accuracy = 98.24228028503563
epoch 4 / 10, Adamax accuracy = 98.29770387965162
epoch 5 / 10, Adamax accuracy = 98.33729216152018
epoch 6 / 10, Adamax accuracy = 98.37688044338876
epoch 7 / 10, Adamax accuracy = 98.49564528899447
epoch 8 / 10, Adamax accuracy = 98.27395091053049
epoch 9 / 10, Adamax accuracy = 98.44813935075217
epoch 10 / 10, Adamax accuracy = 98.36104513064133
---------------------------------
epoch 1 / 10, NAdam accuracy = 95.7957244655582
epoch 2 / 10, NAdam accuracy = 95.8590657165479
epoch 3 / 10, NAdam accuracy = 96.88836104513065
epoch 4 / 10, NAdam accuracy = 96.56373713380839
epoch 5 / 10, NAdam accuracy = 96.82501979414093
epoch 6 / 10, NAdam accuracy = 97.49802058590657
epoch 7 / 10, NAdam accuracy = 97.04671417260491
epoch 8 / 10, NAdam accuracy = 97.30007917656374
epoch 9 / 10, NAdam accuracy = 97.49802058590657
epoch 10 / 10, NAdam accuracy = 97.53760886777513
---------------------------------
epoch 1 / 10, RAdam accuracy = 97.65637371338084
epoch 2 / 10, RAdam accuracy = 97.88598574821853
epoch 3 / 10, RAdam accuracy = 95.40775930324624
epoch 4 / 10, RAdam accuracy = 96.65874901029295
epoch 5 / 10, RAdam accuracy = 97.61678543151227
epoch 6 / 10, RAdam accuracy = 97.70387965162311
epoch 7 / 10, RAdam accuracy = 96.61124307205067
epoch 8 / 10, RAdam accuracy = 97.20506730007918
epoch 9 / 10, RAdam accuracy = 97.41884402216944
epoch 10 / 10, RAdam accuracy = 97.89390340459224
---------------------------------
epoch 1 / 10, RMSprop accuracy = 96.6825019794141
epoch 2 / 10, RMSprop accuracy = 96.3895486935867
epoch 3 / 10, RMSprop accuracy = 96.95170229612035
epoch 4 / 10, RMSprop accuracy = 97.33174980205858
epoch 5 / 10, RMSprop accuracy = 97.6959619952494
epoch 6 / 10, RMSprop accuracy = 97.64053840063342
epoch 7 / 10, RMSprop accuracy = 97.3396674584323
epoch 8 / 10, RMSprop accuracy = 97.60886777513856
epoch 9 / 10, RMSprop accuracy = 97.1021377672209
epoch 10 / 10, RMSprop accuracy = 97.34758511480601
---------------------------------
epoch 1 / 10, Rprop accuracy = 96.06492478226446
epoch 2 / 10, Rprop accuracy = 95.69279493269993
epoch 3 / 10, Rprop accuracy = 94.33887569279493
epoch 4 / 10, Rprop accuracy = 94.57640538400634
epoch 5 / 10, Rprop accuracy = 93.18289786223278
epoch 6 / 10, Rprop accuracy = 92.50197941409343
epoch 7 / 10, Rprop accuracy = 93.11955661124307
epoch 8 / 10, Rprop accuracy = 91.79730799683293
epoch 9 / 10, Rprop accuracy = 91.44893111638956
epoch 10 / 10, Rprop accuracy = 91.97941409342835
```
# Best Per Optimizer Final Results:
```
Rprop accuracy = 96.06492478226446
RMSprop accuracy = 97.6959619952494
RAdam accuracy = 97.89390340459224
NAdam accuracy = 97.53760886777513
Adamax accuracy = 98.49564528899447
AdamW accuracy = 97.61678543151227
Adagrad accuracy = 97.70387965162311
Adadelta accuracy = 96.83293745051465
ASGD accuracy = 96.78543151227237
ADAM accuracy = 97.06254948535233
SGD accuracy = 96.57957244655582
```

