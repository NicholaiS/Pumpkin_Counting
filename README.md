# Pumpkin_Counting
The following is a Mini project of counting pumpkins in a field.

## Segmantation_tests.py
Used for testing and visualizing Colour segmentation methods:
- BGR inRange
- CieLAB inRange
- Distance in BGR space to a reference colour

## Mask.py
Used for a more in-depth understanding of the Distance in BGR space to a reference colour segmentation method.
Using a mask made in GIMP a mean colour value and covariance matrix is found.
The distribution of each colour channel (both in RGB and CieLAB) is then visualized as histograms and saved in the 'Mask data' folder. The mean colour is also saved as a image.

## Chunk_manipulator.py
Contains functions used in Pumpkin_counting.py to manipulate chunks. Both making them and stitching them back together again.

## Image_processing.py
Contains functions used in Pumpkin_counting.py in order to make colour segmentation, feature extraction and saving positions of objects of interest.

## Source.py
Iterates over the .tif file using chunks. Doing colour segmentation, feature extraction and getting position of objects of interest on each chunk.
Then removing duplicates made by splitting an object with a chunk border.
Lastly marking the objects using circles in the entire stitched image. 
