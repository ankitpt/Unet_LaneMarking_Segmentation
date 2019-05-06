from PIL import Image 
import numpy as np
import os
import re
# Read image 


os.chdir("//myhome.itap.purdue.edu/myhome/pate1019/ecn.data/Desktop/DPRG_ECN/training_new/onboard2")

directory = os.fsencode("//myhome.itap.purdue.edu/myhome/pate1019/ecn.data/Desktop/DPRG_ECN/training_new/onboard2")
k=0
fil=0
ls=[47,36,32,31,30,29,12,11,1,0]

filelist=os.listdir(directory)
filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0]))

for file in filelist:
    
    src= os.fsdecode(file)
    no=(src[:-4])
    dst=no+'n'+'.png'
    os.rename(src, dst) 

filelist=os.listdir(directory)
filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0][:-1]))

for file in filelist:
    
    src = os.fsdecode(file)
    print("Processing file ",src)
    no=int(float((src[:-5])))
    if(no in ls):
        os.remove(src)
        continue
    else:
        dst=str(k)+".png"
        os.rename(src, dst) 
        k=k+1
    