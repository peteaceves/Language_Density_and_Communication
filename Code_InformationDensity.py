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
from gensim.models import Word2Vec 
from gensim.models.doc2vec import Doc2Vec
import gensim

print (gensim.__version__)

def clean_line(line):
    data = line.lower()                                      # Turn to lowercase
    exclude = set(string.punctuation)                        # Creates set of punctuation strings to remove
    exclude.update(['”', "“", "’"])                          # Adds these symbols to the set of excluded strings
    data = ''.join(ch for ch in data if ch not in exclude)   # Remove punctuation
    data = u''.join([i for i in data if not i.isnumeric()])  # Removes numeric strings
    return data

def encode(symb2freq):
    """Huffman encode the given dict mapping symbols to weights"""
    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def huff(symb2freq):
    huffmansymbols = ""
    huffw = encode(symb2freq)
    for p in huffw:
        huffmansymbols += p[1] * symb2freq[p[0]]
    huff = len(huffmansymbols) / 8
    return huff
    
# Processes gzip line by line for sentence level measures

infolder = 'FILENAME_LOCATION'

for file in os.listdir(infolder):

    outfilename = 'FILENAME_LOCATION' + str(file) + '.csv'
    corpus      = file.rsplit('_', 1)[0]
    
    start = time.time()
    
    with open(outfilename, 'w', newline='') as out:

        spamwriter = csv.writer(out, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        fieldnames = (corpus, "lang", "s", "w", "rw", "t", "rt", "huff", "rhuff")
        spamwriter.writerow(fieldnames)

        langs1 = (re.split('-|\.', file)[0])
        langs1 = (re.split('_', langs1)[-1])
        langs2 = (re.split('-|\.', file)[1])

        if langs1 == "en":
            lang1 = langs1
            lang2 = langs2
        else:
            lang1 = langs2
            lang2 = langs1

        w1 = 0
        w2 = 0
        
        s1 = 0
        s2 = 0
        
        fname = infolder + file
        
        # Word-level dictionaries
        symb2freq1 = defaultdict(int)
        symb2freq2 = defaultdict(int)
        
        # Character-level dictionaries

        with gzip.open(fname,'r') as f:        
            
            context = etree.iterparse(f, tag="tuv")
            context = iter(context)
            event, root = next(context)
            
            for event, element in context:
                
                if element.attrib['{http://www.w3.org/XML/1998/namespace}lang'] == lang1:
                    
                    for child in element:
                        s1 += 1
                        sent1 = clean_line(child.text)
                        wrds1 = sent1.split()
                        w1   += len(wrds1)
                        for word in wrds1:
                            symb2freq1[word] += 1
                        
                elif element.attrib['{http://www.w3.org/XML/1998/namespace}lang'] == lang2:
                    
                    for child in element:
                        s2 += 1
                        sent2 = clean_line(child.text)
                        wrds2 = sent2.split()
                        w2   += len(wrds2)
                        for word in wrds2:
                            symb2freq2[word] += 1
                    
                else:
                    continue
                
                element.clear()
                root.clear()
                for ancestor in element.xpath('ancestor-or-self::*'):
                    while ancestor.getprevious() is not None:
                        del ancestor.getparent()[0]                        

            # 1. Huffsize
            
            t1 = len(symb2freq1)
            huff1 = huff(symb2freq1)

            # 2. Huffsize
            t2 = len(symb2freq2)
            huff2 = huff(symb2freq2)
            
            # RATIO measures
            
            rw1 = w1 / w1
            rw2 = w2 / w1
            
            rt1 = t1 / t1
            rt2 = t2 / t1
            
            rhuff1 = huff1 / huff1
            rhuff2 = huff2 / huff1
            
            # Write measures out to csv

            spamwriter.writerow([corpus]+[lang1]+[s1]+[w1]+[rw1]+[t1]+[rt1]+[huff1]+[rhuff1])
            spamwriter.writerow([corpus]+[lang2]+[s2]+[w2]+[rw2]+[t2]+[rt2]+[huff2]+[rhuff2])
            
            del (context)

    print("corpus: ", corpus, " completed in: ", time.time() - start, " seconds")
    process = psutil.Process()
    print(process.memory_info().rss / 1073741824)  # in GB 
    
