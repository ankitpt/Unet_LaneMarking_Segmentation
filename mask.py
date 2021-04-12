from PIL import Image 
import numpy as np
import os
import re
# Read image
def mask(filename,k):
    
    img = Image.open(filename) 
    img2=np.asarray(img)
    img2=img2*255
    img2 = Image.fromarray(img2.astype(np.uint8))
    img2=img2.resize((256,256))
    nfilename=str(k)+".png"
    os.chdir("//myhome.itap.purdue.edu/myhome/pate1019/ecn.data/Desktop/DPRG_ECN/training_new/onboard")
    img2.save(nfilename,"PNG")
    os.chdir("//myhome.itap.purdue.edu/myhome/pate1019/ecn.data/Desktop/PixelLabelData_2")
 



k=1
os.chdir("//myhome.itap.purdue.edu/myhome/pate1019/ecn.data/Desktop/PixelLabelData_2")

directory = os.fsencode("//myhome.itap.purdue.edu/myhome/pate1019/ecn.data/Desktop/PixelLabelData_2")

filelist=os.listdir(directory)
filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0][6:]))

for file in filelist:
    
    filename = os.fsdecode(file)
    print("Processing file ",filename)
    mask(filename,k)
    k=k+1
    