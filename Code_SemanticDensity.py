import xml.etree.ElementTree as ET
from lxml import etree
import requests 
import numpy as np 
import pandas 
import os 
import os.path 
import pandas as pd
import gzip
import shutil
import time
import string
import numpy as np
from scipy.integrate import simps
import math
import csv
import re
from collections import defaultdict
from collections import Counter
from heapq import heappush, heappop, heapify
import codecs
import multiprocessing
import random
from scipy.stats import variation
import scipy
from math import log2
from pathlib import Path
import sys
import psutil

# Working with strings
import string
import re
import nltk
import nltk.data
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#print (gensim.__version__)

def clean_line(line):
    data = line.lower()                                      # Turn to lowercase
    exclude = set(string.punctuation)                        # Creates set of punctuation strings to remove
    exclude.update(['', "", ""])                          # Adds these symbols to the set of excluded strings
    data = ''.join(ch for ch in data if ch not in exclude)   # Remove punctuation
    data = u''.join([i for i in data if not i.isnumeric()])  # Removes numeric strings
    return data

class sent_iter1():
    def __iter__(self):
        with gzip.open(fname,'r') as f:        
            context = etree.iterparse(f, tag="tuv")
            context = iter(context)
            event, root = next(context)
            for event, element in context:
                if element.attrib['{http://www.w3.org/XML/1998/namespace}lang'] == lang1:
                    for child in element:
                        sent1 = clean_line(child.text)
                        wrds1 = sent1.split()
                        yield wrds1

class sent_iter2():
    def __iter__(self):
        with gzip.open(fname,'r') as f:        
            context = etree.iterparse(f, tag="tuv")
            context = iter(context)
            event, root = next(context)
            for event, element in context:
                if element.attrib['{http://www.w3.org/XML/1998/namespace}lang'] == lang2:
                    for child in element:
                        sent1 = clean_line(child.text)
                        wrds1 = sent1.split()
                        yield wrds1

def dist_and_cv(model): # Calculates the average sim and coeff of var for 10,000 random pairs of words
    distances = []
    word_indexes = len(model.wv.vocab.items()) - 1
    for i in range(10000): 
        rand1 = model.wv.index2entity[random.randrange(0, word_indexes)]
        rand2 = model.wv.index2entity[random.randrange(0, word_indexes)]
        dist = model.wv.similarity(rand1, rand2) + 1
        distances.append(dist)
        
    mean_dist = float(sum(distances)) / float(len(distances))
    cv        = variation(distances)

    return mean_dist, cv

# Model parameters

VectorSize = 300
Epochs     = 10
Window     = 5

infolder = 'FILENAME_LOCATION'
fil = sys.argv[1]               #read the filename from command arg (provided in the .sh)
infile = infolder + fil                   

outfilename = 'FILENAME_LOCATION' + str(fil) + '.csv'

corpus      = fil.rsplit('_', 1)[0]
    
start = time.time()

with open(outfilename, 'w', newline='') as out:

    spamwriter = csv.writer(out, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    fieldnames = (corpus, "lang", "dist", "rdist", "cv", "rcv")
    spamwriter.writerow(fieldnames)

    langs1 = (re.split('-|\.', fil)[0])
    langs1 = (re.split('_', langs1)[-1])
    langs2 = (re.split('-|\.', fil)[1])

    #if langs1 == "en":   # For oo3 files, which have a funky name for english inside the document
    #    lang1 = "en_GB"
    #    lang2 = langs2

    if langs1 == "en":
        lang1 = langs1
        lang2 = langs2
    else:
        lang1 = langs2
        lang2 = langs1

    fname = infolder + fil

    # Language1 = English
    
    sents1 = sent_iter1()
    model1 = Word2Vec(size = VectorSize, window = Window)
    model1.build_vocab(sentences = sents1)   # build the vocabulary
    model1.train(sentences = sents1, epochs = Epochs, total_examples = model1.corpus_count)
    
    modName1 = str(fil) + "_" + lang1 + "_" + str(VectorSize) + "-" + str(Epochs) + "-" + str(Window) + "-" + ".model"
    
    model1.save('FILENAME_LOCATION' + modName1)

    # Language2 = Other

    sents2 = sent_iter2()
    model2 = Word2Vec(size = VectorSize, window = Window)
    model2.build_vocab(sentences = sents2)   # build the vocabulary
    model2.train(sentences = sents2, epochs = Epochs, total_examples = model2.corpus_count)
    
    modName2 = str(fil) + "_" + lang2 + "_" + str(VectorSize) + "-" + str(Epochs) + "-" + str(Window) + "-" + ".model"
    model2.save('FILENAME_LOCATION' + modName2)

    # Embedding Measures

    result1    = dist_and_cv(model1)
    mean_dist1 = result1[0]
    cv1        = result1[1]

    result2    = dist_and_cv(model2)
    mean_dist2 = result2[0]
    cv2        = result2[1]

    # Ratio Measures

    rdist1 = mean_dist1 / mean_dist1
    rdist2 = mean_dist2 / mean_dist1

    rcv1   = cv1 / cv1
    rcv2   = cv2 / cv1

    # Write measures out to csv

    spamwriter.writerow([corpus]+[lang1]+[mean_dist1]+[rdist1]+[cv1]+[rcv1])
    spamwriter.writerow([corpus]+[lang2]+[mean_dist2]+[rdist2]+[cv2]+[rcv2])

    print("corpus: ", corpus, " completed in: ", time.time() - start, " seconds")
    process = psutil.Process()
    print(process.memory_info().rss / 1073741824)  # in GB 

out.close()

