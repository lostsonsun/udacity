import argparse

import numpy as np
import time
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser(description='Predict the class for flowers.')
parser.add_argument('input', dest='checkpoint')
parser.add_argument('--arch', dest='arch')
parser.add_argument('--image_path', dest='img')
parser.add_argument('--category_names', dest='cat_names')
parser.add_argument('--topk', dest='topk')
parser.add_argument('--gpu', dest='gpu')
args = parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if args.arch == 'dense':
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model, optimizer

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = image.resize((256, 256))
    image = image.crop((16, 16, 16+224, 16+224))
    np_image = np.array(image)
    
    np_image = (np_image/255.0*0.99) + 0.01
    
    mean = np.array([0.485, 0.456, 0.406])
    sdv = np.array([0.229, 0.224, 0.225])
    np_image = (np_image-mean)/sdv
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, cuda, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(image_path)
    inputs = torch.from_numpy(process_image(image))
    inputs = inputs.unsqueeze(0).float()
    if cuda == 'gpu' and torch.cuda.is_available():
        inputs = inputs.to('cuda')
    else:
        inputs = inputs.to('cpu')
    
    outputs = model(inputs)
    ps, classes_index = outputs.topk(topk)
    ps = torch.exp(ps.data.cpu()).numpy()[0]
    classes_index = classes_index.data.cpu().numpy()[0]
    
    class_to_idx_keys = model.class_to_idx.keys()
    key_list = []
    for key in class_to_idx_keys:
        key_list.append(key)
        
    classes = []
    for c in classes_index:
        classes.append(key_list[c])
    
    return ps, classes

model, optimizer = load_checkpoint(args.checkpoint)

cuda = args.gpu
if cuda == 'gpu' and torch.cuda.is_available():
    model.to('cuda')
else:
    model.to('cpu')

probs, classes = predict(args.img, model, cuda, args.topk)

with open(args.cat_names, 'r') as f:
    cat_to_name = json.load(f)

classes_name = []
for c in classes:
    classes_name.append(cat_to_name[c])

print('Top 5 probability are:')
print(probs)
print('Top 5 classes are:')
print(classes)

