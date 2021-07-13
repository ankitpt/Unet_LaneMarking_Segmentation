from PIL import Image 
import numpy as np
import os
import re
# Read image
def mask(filename,k,source_dir,target_dir):
    
    img = Image.open(filename) 
    img2=np.asarray(img)
    img2=img2*255
    img2 = Image.fromarray(img2.astype(np.uint8))
    img2=img2.resize((256,256))
    nfilename=str(k)+".png"
  
    os.chdir(target_dir)
    
    img2.save(nfilename,"PNG")
    os.chdir(source_dir)
 



k=0
cwd=os.getcwd()
source_dir="Image_labeler_output"
target_dir="Processed_Image_labeler_output"


abs_original_label_dir=os.path.join(os.sep,cwd,source_dir)
abs_processed_label_dir=os.path.join(os.sep,cwd,target_dir)

os.chdir(abs_original_label_dir)

directory = os.fsencode(abs_original_label_dir)

filelist=os.listdir(directory)
filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0][6:]))

#filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f.decode()))))

for file in filelist:
    
     
    filename = os.fsdecode(file)
    #filename=str(file)+".png"
    print("Processing file ",filename)
    mask(filename,k,abs_original_label_dir,abs_processed_label_dir)
    k=k+1