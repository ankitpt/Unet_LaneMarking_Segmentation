# Lidar_Lane_Width

This work aims at estimating lane width along a highway from its Lidar point cloud data.

1) First, point cloud blocks of length 12.8 along road surface are extracted
2) They are then converted into intensity images which are then fed to U-net trained to segment road markings from road surface
3) The resultant image of U-net is then used to detect ambiguous road markings and subsequently estimate lane widths at regular intervals
