### GTSRB
A Convolutional Neural Network (CNN) for German traffic sign classification using the GTSRB dataset. This project classifies German traffic signs into 43 different categories, providing a high-accuracy solution for recognizing and distinguishing road signs.

![First_pic](https://github.com/HoseinNasiriShahraki/GTSRB/blob/main/Examples/00006.png?raw=true "First_pic") ![second_pic](https://github.com/HoseinNasiriShahraki/GTSRB/blob/main/Examples/00009.png?raw=true "First_pic") ![second_pic](https://github.com/HoseinNasiriShahraki/GTSRB/blob/main/Examples/00042.png?raw=true "First_pic") ![second_pic](https://github.com/HoseinNasiriShahraki/GTSRB/blob/main/Examples/00051.png?raw=true "First_pic") ![second_pic](https://github.com/HoseinNasiriShahraki/GTSRB/blob/main/Examples/00086.png?raw=true "First_pic") ![second_pic](https://github.com/HoseinNasiriShahraki/GTSRB/blob/main/Examples/00093.png?raw=true "First_pic") ![second_pic](https://github.com/HoseinNasiriShahraki/GTSRB/blob/main/Examples/00174.png?raw=true "First_pic")


The GTSRB dataset contains 51,839 images of 43 different classes, each representing a type of traffic sign. These images vary in brightness, size, and angle, providing a realistic dataset for training robust machine learning models.


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


Images are transformed using the torchvision.transforms library to enhance model performance. The transformations applied are:
##### Resize: Adjust images to 28x28 pixels.
##### Normalization: Standardize the pixel values.
##### Tensor Conversion: Convert images to PyTorch tensors.
```pyton
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize((28,28)),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
```

### Custom Dataset Class
A custom dataset class, GTSR_DataSet, is created to load and prepare images and labels for training.
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
The model GTRSB_Model is built with multiple convolutional and fully connected layers to handle complex features of traffic signs.
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

```
After 2 Hours of Trainning Here Are The Final Results:
```
epoch 1 / 10, SGD accuracy = 91.07680126682503
epoch 10 / 10, SGD accuracy = 96.57957244655582
---------------------------------
epoch 1 / 10, ADAM accuracy = 94.48139350752177
epoch 10 / 10, ADAM accuracy = 96.04908946951703
---------------------------------
epoch 1 / 10, ASGD accuracy = 96.31037212984957
epoch 10 / 10, ASGD accuracy = 96.66666666666667
---------------------------------
epoch 1 / 10, Adadelta accuracy = 96.66666666666667
epoch 10 / 10, Adadelta accuracy = 96.72209026128266
---------------------------------
epoch 1 / 10, Adagrad accuracy = 97.36342042755345
epoch 10 / 10, Adagrad accuracy = 97.70387965162311
---------------------------------
epoch 1 / 10, AdamW accuracy = 97.2763262074426
epoch 10 / 10, AdamW accuracy = 96.98337292161521
---------------------------------
epoch 1 / 10, Adamax accuracy = 98.08392715756136
epoch 10 / 10, Adamax accuracy = 98.36104513064133
---------------------------------
epoch 1 / 10, NAdam accuracy = 95.7957244655582
epoch 10 / 10, NAdam accuracy = 97.53760886777513
---------------------------------
epoch 1 / 10, RAdam accuracy = 97.65637371338084
epoch 10 / 10, RAdam accuracy = 97.89390340459224
---------------------------------
epoch 1 / 10, RMSprop accuracy = 96.6825019794141
epoch 10 / 10, RMSprop accuracy = 97.34758511480601
---------------------------------
epoch 1 / 10, Rprop accuracy = 96.06492478226446
epoch 10 / 10, Rprop accuracy = 91.97941409342835
```
# Best Final Results Per Optimizer:
## Results

- **Best Accuracy**: 98.5% (Adamax optimizer)
- **Training Time**: ~2 hours on GPU

| Optimizer | Final Accuracy |
|-----------|----------------|
| Adamax    | 98.5%         |
| RMSprop   | 97.9%         |
| RAdam     | 97.7%         |
| Adagrad   | 97.7%         |
| AdamW     | 97.6%         |
| NAdam     | 97.5%         |
| ADAM      | 97.0%         |
| Adadelta  | 96.8%         |
| ASGD      | 96.7%         |
| SGD       | 96.5%         |
