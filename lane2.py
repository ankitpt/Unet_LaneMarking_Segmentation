tile=input("Which tile to process? ")
import time

start = time.time()

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import math

def get_dis(c1,c2,vx,vy):
    
    dis=abs(vx*c1-vx*c2)/np.sqrt(vx*vx+vy*vy)
    return dis

def project(pnts,vx,vy,xo,yo):
    
    pro_pnts=np.zeros((len(pnts),2))
    k=0
    for pnt in pnts:
        
        x=pnt[0][0]
        y=pnt[0][1]
        
        p=((x-xo)*vx+(y-yo)*vy)/(vx**2+vy**2)
        xp=p*vx+xo
        yp=p*vy+yo
        
       # print(x,y,xp,yp)
        pro_pnts[k,:]=np.reshape((xp,yp),(1,2))
        k=k+1
     
    return pro_pnts


if os.path.exists("C:/Users/17657/Desktop/DPRG/lw/lw_"+tile+".txt"):
  os.remove("C:/Users/17657/Desktop/DPRG/lw/lw_"+tile+".txt")
else:
  pass

if os.path.exists("C:/Users/17657/Desktop/DPRG/ambiguous/ambiguous_"+tile+".txt"):
  os.remove("C:/Users/17657/Desktop/DPRG/ambiguous/ambiguous_"+tile+".txt")
else:
  pass


os.chdir("C:/Users/17657/Desktop/DPRG/sf")
scale=open("sf_"+tile+".txt","r")

os.chdir("C:/Users/17657/Desktop/DPRG/georef")
geo=open("georef_"+tile+".txt","r")

sfs=np.zeros((100,2))
geos=np.zeros((100,4))
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

os.chdir("C:/Users/17657/Desktop/DPRG")
wide=list()
var=list()


for img_k in [45]:
    
    
    img=cv2.imread(str(img_k)+"_predict.png",0)        
    nimg=np.zeros((256,256))
    #fitting line to trj points
    
    img_trj=cv2.imread(str(img_k)+".png",0)
    trj=np.where(img_trj==255)
    trj_pts=list()
    
    for (x,y) in zip(trj[0],trj[1]):
        trj_pts.append([y,x])
    
    [vx,vy,x,y] = cv2.fitLine(np.array([trj_pts]), cv2.DIST_L2,0,0.01,0.01)
    
    
    #extracting road lane perpendicular to trajectory line
    for y1 in range(0,256):
        
        c_cept=y-(vy/vx)*x
        x1=(y1-(c_cept))*(vx/vy)
        dist=np.sqrt(vx*vx+vy*vy)
        dx=vx/dist
        dy=vy/dist
        x_right=x1+(55/sfs[img_k,0])*abs(dy)
        x_left=x1-(50/sfs[img_k,0])*abs(dy)
        xnew=np.arange(int(np.round(x_left)),int(np.round(x_right))+1,1)
        nimg[y1,xnew]=img[y1,xnew]
    
    
    #cv2.line(img_trj,(y1,x1),(y2,x2),(255,0,0),2)
    
    
    
    
    
    nimg = nimg.astype(np.uint8)
    image, contours, hierarchy = cv2.findContours(nimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    
    if(len(contours)<=1):
        wide.append(img_k)
        continue
    
    
    rows,cols = nimg.shape[:2]
    rgb = cv2.cvtColor(nimg,cv2.COLOR_GRAY2RGB)
    k=0
    
    lftm=255
    rtm=0
    tpm1=255
    tpm2=255
    
    left_sign=np.sign(255-(vy/vx)*0-c_cept)
    right_sign=np.sign(0-(vy/vx)*255-c_cept)
    #finding left and rightmost lane marker
    for contour in contours:
        
        A=cv2.contourArea(contour)
        P=cv2.arcLength(contour,True)+0.000001
        W=(P-np.sqrt(P*P-16*A))/4
        L=(P+np.sqrt(P*P-16*A))/4
        
       #checking A/P ratio to see if the cluster is invalid
        if(math.isnan(W) or math.isnan(L) or (W<2 or L<20)):
            k=k+1
            #print(cv2.contourArea(contour)/cv2.arcLength(contour,True))
            print(img_k)
            print(W,L)
            print("oh")
            continue
        
       
            
        mean=np.mean(contour, axis=0)
        leftmost = tuple(contour[contour[:,:,0].argmin()][0])
        rightmost = tuple(contour[contour[:,:,0].argmax()][0])
        topmost = tuple(contour[contour[:,:,1].argmin()][0])
    
        
        #if(leftmost[0]<lftm ):
        if(mean[0][0]<lftm and np.sign(mean[0][1]-(vy/vx)*mean[0][0]-c_cept)==left_sign):   
            cl=k
            #lftm=leftmost[0]
            lftm=mean[0][0]
            lftm_y=mean[0][1]
        #if(rightmost[0]>rtm):
        elif(mean[0][0]>rtm and np.sign(mean[0][1]-(vy/vx)*mean[0][0]-c_cept)==right_sign):
            
            cr=k
            rtm=mean[0][0]
            rtm_y=mean[0][1]
        
        k=k+1    
    
    
    m=vy/vx
    c1=lftm_y-m*lftm
    c2=rtm_y-m*rtm
    dis=get_dis(c1,c2,vx,vy)
    
    if (dis<50/sfs[img_k,0]):
        
        if(cl==cr):
            
            var.append(img_k)
            var.append(P/A)
            var.append(A)
            var.append(" ")
            
        else:
            wide.append(img_k)
            print("Left and right most markers are too close, thus potential wide lane in image ",img_k)
            continue
        
        
    
    
    k=0    
    
    lmean=np.mean(contours[cl], axis=0)
    rmean=np.mean(contours[cr], axis=0)
    [vx_lref,vy_lref,x_lref,y_lref] = cv2.fitLine(contours[cl], cv2.DIST_L2,0,0.01,0.01)
    [vx_rref,vy_rref,x_rref,y_rref] = cv2.fitLine(contours[cr], cv2.DIST_L2,0,0.01,0.01)
    #Finding right and left grouping of clusters
    cleft=list()
    cright=list()
    
    
    for contour in contours:
        
        A=cv2.contourArea(contour)
        P=cv2.arcLength(contour,True)+0.000001
        W=(P-np.sqrt(P*P-16*A))/4
        L=(P+np.sqrt(P*P-16*A))/4
        if(math.isnan(W) or math.isnan(L) or (W<2 or L<20)):
            k=k+1
            continue
        
        if(np.all(contour==contours[cl]) or np.all(contour==contours[cr])):
            
            if(k==cl):
                
                cleft.append(k)
            else:
                cright.append(k)
                
            
            k=k+1
            #[vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
           # pro_pnts=project(contour,vx,vy,x,y)
            #for pnt in pro_pnts:
                
             #   cv2.circle(rgb,(int(round(pnt[0])),int(round(pnt[1]))), 0, (0,255,0), -1)
            
            
            continue
        
        
        #
        cmean=np.mean(contour, axis=0)
        
        vl=(cmean-lmean).flatten()
        vr=(cmean-rmean).flatten()
        v_lref=np.array((vx_lref,vy_lref)).flatten()
        v_rref=np.array((vx_rref,vy_rref)).flatten()
        langle = (180/np.pi)*(np.arccos(np.dot(v_lref, vl) / (np.linalg.norm(v_lref) * np.linalg.norm(vl))))
        rangle = (180/np.pi)*(np.arccos(np.dot(v_rref, vr) / (np.linalg.norm(v_rref) * np.linalg.norm(vr))))
        
        
        if(langle>90):
            langle=180-langle
            
        if(rangle>90):
            rangle=180-rangle
        
        if(langle<4):
            
            cleft.append(k)
            print("The contour ",k," belongs to left side",langle)
           # pro_pnts=project(contour,vx,vy,x,y)
            
           # for pnt in pro_pnts:
                
            #    cv2.circle(rgb,(int(round(pnt[0])),int(round(pnt[1]))), 0, (0,255,0), -1)
    
        elif(rangle<4):
            
            cright.append(k)
            print("The contour ",k," belongs to right side",rangle)
            #pro_pnts=project(contour,vx,vy,x,y)
            
            #for pnt in pro_pnts:
                
             #   cv2.circle(rgb,(int(round(pnt[0])),int(round(pnt[1]))), 0, (0,255,0), -1)
    
        else:
            
            
            os.chdir("C:/Users/17657/Desktop/DPRG/ambiguous")
            tmean=np.mean(contour, axis=0)
            print( tmean[0],img_k,file=open("ambiguous_"+tile+".txt", "a"))
            print("The contour ",k," is ambiguous",langle,rangle)
            #emp.append(img_k)
            os.chdir("C:/Users/17657/Desktop/DPRG")
           # pro_pnts=project(contour,vx,vy,x,y)
            
           # for pnt in pro_pnts:
                
            #    cv2.circle(rgb,(int(round(pnt[0])),int(round(pnt[1]))), 0, (255,0,0), -1)
            
            
        
        k=k+1
    
    lp=list()
    rp=list()
    
    #fitting line to left and right side clusters
    for i in cleft:
        
        for pnt in contours[i]:
            
            lp.append(pnt)
            
    for i in cright:
        
        for pnt in contours[i]:
            
            rp.append(pnt)
    
    if(len(lp) == 0 or len(rp)==0):
        
        print("Lane in image ",img_k," is too wide to detect markings on either left or right side")
        continue
        
    [vxl,vyl,xl,yl] = cv2.fitLine(np.array(lp), cv2.DIST_L2,0,0.01,0.01)   
    [vxr,vyr,xr,yr] = cv2.fitLine(np.array(rp), cv2.DIST_L2,0,0.01,0.01)   
    
    pro_pnts=project(np.array(lp),vxl,vyl,xl,yl)
            
    for pnt in pro_pnts:
        
        cv2.circle(rgb,(int(round(pnt[0])),int(round(pnt[1]))), 0, (255,0,0), -1)
    
    pro_pnts=project(np.array(rp),vxr,vyr,xr,yr)
            
    for pnt in pro_pnts:
        
        cv2.circle(rgb,(int(round(pnt[0])),int(round(pnt[1]))), 0, (255,0,0), -1)
    
    
    os.chdir("C:/Users/17657/Desktop/DPRG/lw")
    ys=range(5,255,5)
    
    for y in ys:
        
        x_lt=(y-(yl-(vyl/vxl)*xl))*(vxl/vyl)
        X=geos[img_k,0]+0.05*sfs[img_k,0]*x_lt
        Y=geos[img_k,1]+0.05*sfs[img_k,1]*y
        a=vyr
        b=-vxr
        c=vxr*yr-vyr*xr
        lane_width=(abs(a*x_lt+b*y+c)/np.sqrt(a*a+b*b))*0.05*(sfs[img_k,0])*3.28084
        print("Estimated lane width is, ",lane_width," feet")
        print( lane_width[0],img_k,X[0],Y,file=open("lw_"+tile+".txt", "a"))
    
    os.chdir("C:/Users/17657/Desktop/DPRG")
    
    plt.imshow(rgb,cmap='gray')
    os.chdir("C:/Users/17657/Desktop/DPRG/try")
    cv2.imwrite("try"+str(img_k)+".png",rgb)
    
    os.chdir("C:/Users/17657/Desktop/DPRG")
    
if(len(wide)!=0):
    
    print("Lane is too wide(>15feet) in tile, ",tile," in following images, ",wide)

if(len(var)!=0):
    
    print("Consider varying A/P or A threshold, present values in order of image, A/P and A are ",var)

scale.close()
plt.imshow(rgb,cmap='gray')
end = time.time()

print(end - start)