import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

def clean(img):
    
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    num_c=len(contours)
    
    for k in range(num_c):
        
        rect = cv2.minAreaRect(contours[k])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        temp=cv2.fillPoly(img, [box], [255,255,255])
        
    #cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
    #plt.imshow(img)
    return temp





cwd=os.getcwd()
source_dir="Automated_labels_raw"
target_dir="Automated_labels_bbox"


abs_original_label_dir=os.path.join(os.sep,cwd,source_dir)
abs_processed_label_dir=os.path.join(os.sep,cwd,target_dir)

os.chdir(abs_original_label_dir)

directory = os.fsencode(abs_original_label_dir)

filelist=os.listdir(directory)
filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0]))

k=0
for file in filelist:

    
    filename = os.fsdecode(file)
    
#    if(filename=="74.png" or filename=="75.png"):
 #       continue
    
    img=cv2.imread(filename,0)
#    img=clean(img)
    #kernel = np.ones((3,3),np.uint8)
    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    if True:
       print(f"Processing {filename}") 
       img=clean(img)
       kernel = np.ones((3,3),np.uint8)
       img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
       
       img = Image.fromarray(img.astype(np.uint8))
       
       
       plt.imshow(img)
    else:
      k=k+1
      continue
    
    os.chdir(abs_processed_label_dir)
    filename=str(k)+'.png'
    img.save(filename,"PNG")
    k=k+1
    os.chdir(abs_original_label_dir)

    
    
    


    
    