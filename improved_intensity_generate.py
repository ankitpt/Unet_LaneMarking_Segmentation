import time
start = time.time()
import numpy as np
import os
from PIL import Image

def point_to_image(filename,num,point_cloud_dir,intensity_image_dir):

    #Reading the point cloud file. The size of the "data" variable depends on the maximum 
    #lines expected in a point cloud text file
    data=np.zeros([12000000,4])
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
    
    #Finding bounds of the point cloud file
    id_xmin=np.argmin(data[:,0])    
    x_min,y_xmin=data[id_xmin,0:2]
    
    id_ymin=np.argmin(data[:,1])    
    x_ymin,y_min=data[id_ymin,0:2]
    
    id_ymax=np.argmax(data[:,1])    
    x_ymax,y_max=data[id_ymax,0:2]
    
    
    id_xmax=np.argmax(data[:,0])    
    x_max,y_xmax=data[id_xmax,0:2]
    
    #Set threhsolding percentile. Here it is 95 which translates to 
    #5th percentile threhsolding in our work
    perc_threshold=95
    
    #Level 1 intensity enhancement:Threhsolding the intensity data in point cloud
    #with the chosen threshold value
    perc=np.percentile(data[:,3],perc_threshold)
    print(f"{perc_threshold} percentile threshold for this block is {perc}")
    intens=data[:,3]
    intens[intens > perc] = 255
    data[:,3]=intens


    #Cell size for intensity image in meters. here it is 0.05m or 5cm
    cell_siz=0.05

    #Size of the input image based on rows and cols. Here, generating a
    #256X256 image 
    cols=256
    rows=256
    
    #Storing the scale factors along x and y direction for conversion of
    #2D lane marking predictions in 3D point cloud later
    sfx=(int((x_max-x_min)/cell_siz)+1)/256
    sfy=(int((y_max-y_min)/cell_siz)+1)/256

    img=np.zeros([rows,cols])
    
    #This variable will have average z value for each pixel based on 3D point
    #falling within that pixel. To be used later with scale factors above for
    #conversion of image lane marking predictions to 3D points
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
    
    #Finding pixel value for each grid cell by averaging intensity of all
    #points falling inside. Doing same for finding z values in each cell
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

    #Level 2 intensity enhancement:Threhsolding the intensity data in 
    #intensity image with the chosen threshold value    
    img[img >= perc] = 255
    img=np.uint8(img)

    #Saving intensity image in a subfolder in intensity_image_dir
    nfilename=str(num)+".png"
    img2 = Image.fromarray(img.astype(np.uint8))

    os.chdir(intensity_image_dir+"/images/")
    img2.save(nfilename,"PNG")
    print("Image saved,",nfilename)
    
    #Saving files that will be later used to convert image lane marking 
    #predictions to 3D points
    # 1. georef.txt: Contains extrame bounds of the point cloud file: x_min,
    #     x_max, y_min, y_max
    # 2. sf.txt: Contains scale factors for each point cloud file
    # 3. zgrid.txt: Contains z values for each pixel of each intensity image 

    os.chdir(intensity_image_dir+"/data_for_3D/")
    print( x_min,y_min,x_max,y_max,file=open("georef.txt", "a"))
    
    print( sfx,sfy,file=open("sf.txt", "a"))
    
    with open('zgrid.txt','ab') as f:
        np.savetxt(f, z_grid, fmt='%1.4f',delimiter=" ")
    f.close()
    
    os.chdir(point_cloud_dir)

def my_makedir(abs_intensity_image_dir,folder):
  

  if not os.path.exists(os.path.join(os.sep,abs_intensity_image_dir,folder)):
      os.mkdir(os.path.join(os.sep,abs_intensity_image_dir,folder))    
    
    
cwd=os.getcwd()
point_cloud_dir="EB"
intensity_image_dir="EB_intensity_images"
abs_intensity_image_dir=os.path.join(os.sep,cwd,intensity_image_dir)

my_makedir(abs_intensity_image_dir,"")
my_makedir(abs_intensity_image_dir,"images")
my_makedir(abs_intensity_image_dir,"data_for_3D")

abs_point_cloud_dir=os.path.join(os.sep,cwd,point_cloud_dir)
os.chdir(abs_point_cloud_dir)
directory = os.fsencode(abs_point_cloud_dir)
filelist=os.listdir(abs_point_cloud_dir)
filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for file in filelist:
        
        filename = os.fsdecode(file)
        j=int(filename[32:-4])
        print("Processing file ",filename,"in tile")
        point_to_image(filename,j,abs_point_cloud_dir,abs_intensity_image_dir)
        print("")
      

print(f"Took {time.time() - start} seconds to generate intensity images")

#Took 5633.29923415184 seconds to generate intensity images