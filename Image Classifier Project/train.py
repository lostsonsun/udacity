import argparse

import numpy as np
import time
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser(description='Train a flower classifier.')
parser.add_argument('--arch', dest='arch')
parser.add_argument('--data_dir', dest='data_dir', type=str)
parser.add_argument('--learning_rate', dest='lr', default=0.001, type=float)
parser.add_argument('--hidden_units', dest='hu', default=5000, type=int)
parser.add_argument('--epochs', dest='epochs', default=3, type=int)
parser.add_argument('--gpu', dest='gpu')
parser.add_argument('--save_dir', dest='checkpoint')
args = parser.parse_args()

train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'
data_dir = args.data_dir

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])])

train_datasets = datasets.ImageFolder(data_dir+train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(data_dir+valid_dir, transform=valid_transforms)
test_datasets = datasets.ImageFolder(data_dir+test_dir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)

if args.arch == 'dense':
      model = models.densenet121(pretrained=True)
else:
      model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

hidden_units = args.hu
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=.5)),
                          ('fc2', nn.Linear(hidden_units, 1000)),
                           ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=.5)),
                          ('fc3', nn.Linear(1000, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

#训练分类器
criterion = nn.NLLLoss()
lr = args.lr
optimizer = optim.Adam(model.classifier.parameters(), lr)

epochs = args.epochs
print_every = 40
steps = 0

cuda = args.gpu
if cuda == 'gpu' and torch.cuda.is_available():
    model.to('cuda')
else:
    model.to('cpu')

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(train_loader):
        steps += 1
        
        if cuda == 'gpu' and torch.cuda.is_available():
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        else:
            inputs, labels = inputs.to('cpu'), labels.to('cpu')
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            
            model.eval()
            accuracy = 0
            test_loss = 0
            
            for ii, (inputs, labels) in enumerate(valid_loader):
                
                if cuda == 'gpu' and torch.cuda.is_available():
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                else:
                    inputs, labels = inputs.to('cpu'), labels.to('cpu')
                
                output = model.forward(inputs)
                test_loss += criterion(output, labels).data[0]
                
                ## Calculating the accuracy 
                # Model's output is log-softmax, take exponential to get the probabilities
                ps = torch.exp(output).data
                # Class with highest probability is our predicted class, compare with true label
                equality = (labels.data == ps.max(1)[1])
                # Accuracy is number of correct predictions divided by all predictions, just take the mean
                accuracy += equality.type_as(torch.cuda.FloatTensor()).mean()
            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(valid_loader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(valid_loader)))
           
            running_loss = 0
            model.train()
            
def save_checkpoint(model, filepath):
    checkpoint = {'class_to_idx': train_datasets.class_to_idx,
                  'optimizer': optimizer.state_dict(),
                  'classifier': model.classifier,
                  'state_dict': model.state_dict()}
    
    torch.save(checkpoint,filepath)
    return filepath

save_checkpoint(model,args.checkpoint)
