import time

start = time.time()

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import itertools
from PIL import ImageFilter
import cv2
from scipy import interpolate


def clean(img):
    
    image, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    num_c=len(contours)
    
    for k in range(num_c):
        
        rect = cv2.minAreaRect(contours[k])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        temp=cv2.fillPoly(img, [box], [255,255,255])
        
        #rgb = cv2.drawContours(rgb,[box],0,(0,0,255),2)
    return temp    


georef_dir="EB_intensity_images/data_for_3D"
pred_image_dir="testing_images2"


    
#os.chdir("C:/Users/17657/Desktop/DPRG_delphi/asphalt_files/sf")
cwd=os.getcwd()
abs_georef_dir=os.path.join(os.sep,cwd,georef_dir)
abs_pred_image_dir=os.path.join(os.sep,cwd,pred_image_dir)

point_cloud_dir="3D_predictions"
abs_pt_cloud_dir=os.path.join(os.sep,cwd,point_cloud_dir)


os.chdir(abs_georef_dir)
scale=open("sf.txt","r")
geo=open("georef.txt","r")
b = np.loadtxt('zgrid.txt')


sfs=np.zeros((3500,2))
geos=np.zeros((3500,4))

k=0
for line in scale:
    
    
    line=line.strip('\n').split(' ')
    
    sfs[k,:]=[float(s) for s in line]
    k=k+1

sfs=sfs[~np.all(sfs == 0, axis=1)]

k=0
for line in geo:
    
    
    line=line.strip('\n').split(' ')
    
    geos[k,:]=[float(s) for s in line]
    k=k+1

geos=geos[~np.all(geos == 0, axis=1)]

scale.close()
geo.close()


    


os.chdir(abs_pred_image_dir)    
max_img=geos.shape[0]
num=0

directory = os.fsencode(abs_pred_image_dir)

filelist=os.listdir(directory)
filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f.decode()))))
#for img_k in img_list:
for ind in range(0,len(filelist),2): 

    l_mark=None
    filename = os.fsdecode(filelist[ind])
   
    nfilename=filename[:-4]+"_predict.png"
    
    #os.chdir("C:/Users/17657/Desktop/Results_final_img2point/I65_19/test_asphalt") 
    img=cv2.imread(nfilename,0)
    
    l_mark=np.where(img==255)
    
    if(np.shape(l_mark)[1]==0):
        num=num+1
        continue
    

    bs=b[256*num:256*num+256,:]
    count=np.shape(l_mark)[1]
    
    Z=bs  
    Z_nzero=np.where(Z[l_mark[0],l_mark[1]]!=0)
    
    X=geos[num,0]+0.05*sfs[num,0]*l_mark[1]
    X=X[Z_nzero[0]]

    Y=geos[num,1]+0.05*sfs[num,1]*(l_mark[0])
    Y=Y[Z_nzero[0]]

    os.chdir(abs_pt_cloud_dir)
    
    with open('pc_'+filename[:-4]+'.txt','ab') as f:
        
        np.savetxt(f, np.transpose([X,Y,Z[l_mark[0],l_mark[1]][Z_nzero]]), fmt='%.6f %.6f %.3f')

    f.close()   
    num=num+1
    
    os.chdir(abs_pred_image_dir)
