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

 - *improved_intensity_generate.py*: Generate intensity images and georeferencing parameters for point cloud blocks in *EB* folder into *images* and *data_for_3D* subfolder, respectively

### Image labeling

1. Manual Labeling

After generating the intensity images, they need to be labelled at pixel level. This is done by using MATLAB's [Image Labeler tool](https://www.mathworks.com/help/vision/ug/get-started-with-the-image-labeler.html). More specifically, one can use **Label Pixels Using Polygon Tool** where all pixels within polygon can be labelled as lane markings while rest are non-lane marking pixels. The labels are exported in form of an image where lane marking pixels have value 1 while non-lane markings ones have value 0. Thus the image will be dark and we can not verify the correctness of labels. To allow that, the values 1 in labels are converted to 255.

 - *mask.py*: Generate easily visualized labels from MATLAB's Image Labeler tool labels in *Automated_labels_raw* folder and save into *Automated_labels_box* folder

2. Automated labeling

In this procedure, we use the threhsolding-based method described in the paper to generate lane markings directly from point clouds. The regions where this method works well are used to generate intensity images for (raw point cloud block, corresponding lane markings) pair to be used for U-net training. Thereafter, lane marking labels are further processed to create bounding boxes around each lane marking segment present in the image to provide better spatial structure. For this, following script can be used:

- *conv_to_bb.py*: Create bounding box around each lane marking segment in the lane marking intensity images present in *Image_labeler_output* folder into *Processed_Image_labeler_output* folder

Once the intensity images are generated and labelled, they can be divided into training and validation data. The directory structure for the same can be observed in *data* folder. Thereafter below script can be run:

### Training and testing

 - *main.py*: Trains the model by augmenting images in *data/train folder* through various manipulations like flipping, zoom in and out and rotating. At each epoch the model is validated on images in *data/val* folder. Training is stopped based on early stopping criteria.

 - *test.py*: Runs the predictions of the trained model on images in *testing_images* folder. Prediction images are saved in the same folder. Trained model can be found at this link : https://drive.google.com/file/d/1yZHiEQHpf_WQtN-S4W39roDljVdQRA0U/view?usp=sharing
 
### 2D Image prediction to 3D point cloud prediction

- *improved_img2point.py*: Convert 2D image predictions *testing_images2* into 3D point cloud using georeferencing files in *EB_intensity images/data_for_3D*. The point cloud files are saved in *3D_predictions*
