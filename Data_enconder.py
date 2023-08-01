#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:36:27 2023 

@author: nakulkumar
"""

import h5py
import glob
import numpy as np
from PIL import Image

image_root = "./images/"
dirs = ["Mild", "Moderate", "None", "Proliferate", "Severe"]

mild = []
moderate = []
none = []
proliferate = []
severe = []
images = [mild, moderate, none, proliferate, severe]
ids = []
vals = []

for i,path in enumerate(dirs):
    curdir = image_root+path
    image_glob = glob.glob(curdir+'/*.png')
    print("Loading images from "+curdir)
    for image in image_glob:
        img = Image.open(image)
        images[i].append(np.asarray(img, dtype=float))
    image_glob.sort()
    for item in image_glob:
        ID = item.split("/")[3][:-4]
        ids.append(ID)
        vals.append(path)

ids = np.array(ids).T
vals = np.array(vals).T
labels = np.vstack((ids, vals)).T

for i,arr in enumerate(images):
    print("Converting "+dirs[i]+" Data to Numpy Array")
    arr = np.array(arr)
    print("Saving Data ...")
    f = h5py.File("./Data/"+dirs[i]+".h5", "w")
    f.create_dataset("data", data=arr)
    print("Saved.")
    f.close()

print("Saving Labels")
np.savetxt("Data/labels.csv", labels, delimiter=",", fmt="%s")
print("Saved.")

