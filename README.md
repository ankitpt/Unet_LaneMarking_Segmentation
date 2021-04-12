# Unet based lane marking extraction from LiDAR intensity images

This repository contains code for lane marking extraction from intensity images derived from LiDAR point clouds. For a more comprehensive description of the work, here is the link to  corresponding [MDPI paper](https://www.mdpi.com/2072-4292/12/9/1379)

Step by step procedure to use the code is under progress.

### Table of Contents

1. [Project Motivation](#motivation)
2. [File Descriptions](#filedescriptions)

## Project Motivation<a name="motivation"></a>
The goal of the project was to train a U-net model to detect lane markings from LiDAR intensity images generated from LiDAR point cloud.

## File Descriptions <a name="filedescriptions"></a>
*EB* folder consists of blocks of LiDAR point cloud that are 12.8 m in length and 12-16 m wide depending upon the high width.
*EB_intensity images* consist of 2 subfolders:
 - *images*: All intensity images corresponding to the point cloud blocks are generated here
 - *data_for_3D*: This folder contains files with that have georeferencing parameters for converting U-net predicted lane marking image back to 3D point clouds
 
Following python files are updated and ready to be used:

 - *improved_intensity_generate.py*: Generated intensity images and georeferencing parameters for point cloud blocks in *EB* folder into *images* and *data_for_3D* subfolder, respectively

Next update: Label generation for intensity images and keras code for U-net training  
