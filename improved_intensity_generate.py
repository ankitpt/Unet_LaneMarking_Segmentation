import time

start = time.time()

from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import itertools
from PIL import ImageFilter
import cv2
import pickle
import random

def point_to_image(filename,num,tile,point_cloud_dir,intensity_image_dir):

#Reading the point cloud file
    data=np.zeros([12000000,4])
    #data=np.zeros([900000,5])
    k=-1

    with open(filename) as infile:
        
        for line in infile:
            
            if(k==-1):
                k=k+1
                continue
        
            line=line.strip('\n').split('\t')
            line=[float(c) for c in line]
            data[k,:]=line
				
            k=k+1
    
    data=data[0:k,:]
    
#Finding bounds of the file
    id_xmin=np.argmin(data[:,0])    
    x_min,y_xmin=data[id_xmin,0:2]
    
    id_ymin=np.argmin(data[:,1])    
    x_ymin,y_min=data[id_ymin,0:2]
    
    id_ymax=np.argmax(data[:,1])    
    x_ymax,y_max=data[id_ymax,0:2]
    
    
    id_xmax=np.argmax(data[:,0])    
    x_max,y_xmax=data[id_xmax,0:2]
    perc=np.percentile(data[:,3],95)

    print(perc)
    intens=data[:,3]
    intens[intens > perc] = 255
    data[:,3]=intens

    cell_siz=0.05
    cols=256
    rows=256
    sfx=(int((x_max-x_min)/cell_siz)+1)/256
    sfy=(int((y_max-y_min)/cell_siz)+1)/256
    print("No. of rows in this file ",rows)
    print("No. of columns in this file ",cols)

    img=np.zeros([rows,cols])
    z_grid=np.zeros([rows,cols])
    
    #Finding points falling in a grid cell
    k=0
    loc=dict()
    for pt in data:
        
        x=pt[0]
        y=pt[1]
        
        j=int((x-x_min)/(cell_siz*sfx))
        i=int((y-y_min)/(cell_siz*sfy))
        loc.setdefault((i,j),[]).append(k)
        k=k+1
    
    cnt=0;
    
    for i in range(rows):
        for j in range(cols):
            
              try:
                data2=data[loc[i,j]]
                val=np.mean(data2[:,3])
                img[i,j]=val
                z_grid[i,j]=np.mean(data2[:,2])
                cnt=cnt+1
 
              except:
               continue
    
    img[img >= perc] = 255
    img=np.uint8(img)
    nfilename=str(num)+".png"
    
    img2 = Image.fromarray(img.astype(np.uint8))

    os.chdir(intensity_image_dir+"/test_"+tile)
    img2.save(nfilename,"PNG")
    print("Image saved,",nfilename)
    
#    os.chdir(intensity_image_dir+"/georef_"+tile)
#    print( x_min,y_min,x_max,y_max,file=open("georef.txt", "a"))
    
#    os.chdir(intensity_image_dir+"/sf_"+tile)
#    print( sfx,sfy,file=open("sf.txt", "a"))
    
#    os.chdir(intensity_image_dir+"/zgrid_"+tile)
#    with open('zgrid.txt','ab') as f:
#        np.savetxt(f, z_grid, fmt='%1.4f',delimiter=" ")
#    f.close()
    
    os.chdir(point_cloud_dir)

def make_dir(intensity_image_dir,folder,tile):
  
  if not os.path.exists(intensity_image_dir+"/"+folder+"_"+tile):
      os.mkdir(intensity_image_dir+"/"+folder+"_"+tile)    
    
    

tile="EB_norm2"
point_cloud_dir="X:/Common/PWMMS/PWMMS-HA/2020/20200304/reconstruction/RoadSurface/IntensityImage/EB_norm"
intensity_image_dir="C:/Users/17657/Desktop/DPRG_231_EB"

#make_dir(intensity_image_dir,"sf",tile)
#make_dir(intensity_image_dir,"georef",tile)
#make_dir(intensity_image_dir,"zgrid",tile)
make_dir(intensity_image_dir,"test",tile)


os.chdir(point_cloud_dir)
directory = os.fsencode(point_cloud_dir)
filelist=os.listdir(directory)
filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f.decode()))))

for file in filelist:
        
        filename = os.fsdecode(file)
        j=int(filename[24:-4])
        print("Processing file ",filename,"in tile",tile)
        point_to_image(filename,j,tile,point_cloud_dir,intensity_image_dir)
        
      
end = time.time()
print(end - start)

    