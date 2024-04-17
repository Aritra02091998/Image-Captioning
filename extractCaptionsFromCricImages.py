"""
This script is created for:

Extracting Objects from the Images given in the CRIC dataset.

It stores the objects in a list of list for the whole dataset then writes the fetched object names in a text file named "objects_train.txt"
"""
import warnings
warnings.filterwarnings('ignore')
print('Importing packages...')

import os
import re
import torch
import spacy
import json
import math
import sys

# path = './obj_status.txt'
# sys.stdout = open(path, 'w')

from tqdm.auto import tqdm
from PIL import Image as img
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter as Count
from transformers import BlipProcessor, BlipForConditionalGeneration
    

# Setting Up BLIP for Image Captioning

print('Importing BLIP Captioner...')

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")


def getImage(image_path):
    raw_image = img.open(image_path).convert('RGB')    
    return raw_image


# Fetching CRIC Dataset Now

print('\nFetching CRIC Data...')

train_file_path = '../cric/train_questions.json'
val_file_path = '../cric/val_questions.json'
test_file_path = '../cric/test_v1_questions.json'

# Training Set
with open(train_file_path, "r") as file:
     train_json = json.load(file)
        
# Validation Set
with open(val_file_path, "r") as file:
     val_json = json.load(file)
        
# Test Set
with open(test_file_path, "r") as file:
     test_json = json.load(file)


print('\nExtracting Training Data...')


# Extracting Data of Training Set

questionList, answerList, imgList, k_triplet = [],[],[],[]

# verifying
indexToExclude = []

with open('../text_files/error1.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExclude.append(number)
        
with open('../text_files/error2.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExclude.append(number)
        
with open('../text_files/error3.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExclude.append(number)

# Storing Trainig Set into Py Lists

for i in tqdm(range(len(train_json))):
    
    if i in indexToExclude:
        continue
        
    pointer = train_json[i]
    
    questionList.append(pointer['question'])
    answerList.append(pointer['answer'])
    imgList.append(pointer['image_id'])
    k_triplet.append( ' '.join(pointer['sub_graph']['knowledge_items'][0]['triplet']) + '. ' )


"""# Subsetting
questionList = questionList[:20]
imgList = imgList[0:20]
"""
captions = list()
failureCount = 0

print('Captioning Images for the full Set:', len(questionList))

print('Storing Captions into Lists ...')

for i in tqdm(range(len(questionList))):
    
    filepath = '../cric/images/img/'
    imgName = str(imgList[i]) + '.jpg'
    concatedPath = os.path.join(filepath, imgName)
    currentImage = getImage(concatedPath)
    
    try:    
        inputs = processor(currentImage, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)

        currentCaption = processor.decode(out[0], skip_special_tokens=True, max_length = 25)
        
        # print(currentCaption)
        captions.append(currentCaption)
    
    except:
        failureCount += 1
        continue

filename = '../cricImageCaptions/blip_2_captions/blip_image_captions_full_train_set.txt' 
print(f'Writing into file {filename}')

with open(filename, 'w') as file:

    for idx, caption in enumerate(captions):
        file.write(f'{idx}_{caption}\n')
    
print(f'\nFile Stored at {filename}')
print(f'\nCaptioning Failed For {failureCount} Images')
print(f'\nLength of the Captions List is {len(captions)}')
print('\n** Exiting Script **\n')