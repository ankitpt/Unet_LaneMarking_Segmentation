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


#tiles=["11","12","13","14","15","16","17","18","19",
     #  "20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36"]
 #"37","38","39","40",
       #"41","42","43","44","45","46","47","48","49","50","51","52","53","54","55","56"]
#tiles=["12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36"]

#tiles=["11","34","35"]
#tiles=["25"]
tiles=["01"]     
#tiles=["04"]  
for tile in tiles:
    
    #os.chdir("C:/Users/17657/Desktop/DPRG_delphi/asphalt_files/sf")
    os.chdir("C:/Users/17657/Desktop/DPRG_231_SB2/sf")
    scale=open("sf_"+tile+".txt","r")
    
    os.chdir("C:/Users/17657/Desktop/DPRG_231_SB2/georef")
    #os.chdir("C:/Users/17657/Desktop/DPRG_delphi/asphalt_files/georef")
    geo=open("georef_"+tile+".txt","r")
    
    
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
    
  #  if os.path.exists("C:/Users/17657/Desktop/DPRG_delphi/img2point/pc_"+tile+".txt"):
   #   os.remove("C:/Users/17657/Desktop/DPRG_delphi/img2point/pc_"+tile+".txt")
    #else:
     # pass
    
    #if os.path.exists("C:/Users/17657/Desktop/DPRG_delphi/img2point/pc_combined.txt"):
     # os.remove("C:/Users/17657/Desktop/DPRG_delphi/img2point/pc_combined.txt")
    #else:
     # pass
    
    
#    os.chdir("C:/Users/17657/Desktop/DPRG_delphi/files_concrete/zgrid")
#    b = np.loadtxt('zgrid_'+tile+'.txt')

        
    print("Processing tile ",tile)
    
    os.chdir("C:/Users/17657/Desktop/transfer_analysis/test_01_SB2")
             #Delphi_19/test_conc")  
   # max_img=len([name for name in os.listdir('.') if os.path.isfile(name)])
    
    max_img=geos.shape[0]
    num=0
    #img_list=[1676,1677,1678,1679,1726,1727,1728]+list(range(1780,1999))
    #img_list=range(1729,1780)
    
    #os.chdir("C:/Users/17657/Desktop/DPRG_delphi/labels")

    directory = os.fsencode("C:/Users/17657/Desktop/transfer_analysis/test_01_SB2")
                            #Delphi_19/test_conc")
    filelist=os.listdir(directory)
    
     
    
    #filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0][-4:]))
    filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f.decode()))))
    #for img_k in img_list:
    for ind in range(0,len(filelist),2): 
    
        l_mark=None
        filename = os.fsdecode(filelist[ind])
       
        nfilename=filename[:-4]+"_predict.png"
        
        #os.chdir("C:/Users/17657/Desktop/Results_final_img2point/I65_19/test_asphalt") 
        img=cv2.imread(nfilename,0)

        
        #img=cv2.imread(str(img_k)+"_predict.png",0)
        #img=clean(img)        
       # img=np.rot90(np.fliplr(img),2)
        
        l_mark=np.where(img==255)
        
        if(np.shape(l_mark)[1]==0):
            num=num+1
            continue
        
     #   bs=b[256*img_k:256*img_k+256,:]
        bs=b[256*num:256*num+256,:]
    #    bs=np.rot90(np.fliplr(bs),2)
        count=np.shape(l_mark)[1]
        
        #for k in range(0,count):
            
         #   x_cloud=l_mark[1][k]
          #  y_cloud=l_mark[0][k]
           # X=geos[img_k,0]+0.05*sfs[img_k,0]*x_cloud
            #Y=geos[img_k,1]+0.05*sfs[img_k,1]*(255-y_cloud)
            #Z=bs[x_cloud,y_cloud]
            
            #Z=get_Z(X,Y)
        Z=bs  
       # Z_nzero=np.where(Z[255-l_mark[0],l_mark[1]]!=0)    
        Z_nzero=np.where(Z[l_mark[0],l_mark[1]]!=0)
        
      #  X=geos[img_k,0]+0.05*sfs[img_k,0]*l_mark[1]
        X=geos[num,0]+0.05*sfs[num,0]*l_mark[1]
      #  print(X.shape)
        X=X[Z_nzero[0]]
       # print(X.shape)
      #  X=X.reshape((count,1))
       # Y=geos[img_k,1]+0.05*sfs[img_k,1]*(255-l_mark[0])
       # Y=geos[img_k,1]+0.05*sfs[img_k,1]*(l_mark[0])
        Y=geos[num,1]+0.05*sfs[num,1]*(l_mark[0])
        Y=Y[Z_nzero[0]]
        #Y=geos[img_k,1]+0.05*sfs[img_k,1]*(255-l_mark[0])
       # Y=Y.reshape((count,1))
        
        #Z=np.rot90(np.fliplr(bs),2)
        
        #temp=np.where(Z!=0)
   # print(np.shape(temp),temp)
        #xar = np.arange(0, 256)
        #yar = np.arange(0, 256)
        #x_is=temp[0]
        #y_is=temp[1]
    
        #xx, yy = np.meshgrid(xar, yar)
    
        #GD1 = interpolate.griddata((x_is, y_is), Z[temp],(xx, yy),method='nearest')
    #GD2=GD1
        
        
        #Z=Z.reshape((count,1))
        os.chdir("C:/Users/17657/Desktop/DPRG_231_SB2/img2point")
        #Z[255-l_mark[0],l_mark[1]]
        
        with open('pc_'+tile+"_"+filename[:-4]+'.txt','ab') as f:
            
            np.savetxt(f, np.transpose([X,Y,Z[l_mark[0],l_mark[1]][Z_nzero]]), fmt='%.6f %.6f %.3f')
        #     np.savetxt(f, np.transpose([X,Y,Z[255-l_mark[0],l_mark[1]]]), fmt='%.6f %.6f %.3f')
        f.close()   
        
        #np.savetxt('myfile.txt', (X,Y,Z), fmt='%.6f', delimiter=' ', newline=os.linesep)    
        #print("{0:.6f}".format(X),"{0:.6f}".format(Y),"{0:.3f}".format(Z),file=open("pc_"+tile+".txt", "a"))
        num=num+1
        #print(num)
        os.chdir("C:/Users/17657/Desktop/transfer_analysis/test_01_SB2")
                 #Delphi_19/test_conc") 
        
        
    #os.chdir("C:/Users/17657/Desktop/DPRG_delphi/img2point") 
    
