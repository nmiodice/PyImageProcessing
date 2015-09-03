# PyImageProcessing
Some image processing utilities written in Python3

## Why? ##
- I wanted to implement some of my studies in computer vision and machine learning into something that could create art (even if its *bad* art)

## Cool, but what does it do? ##
The code isn't overly complicated, but it does do a few cool things:
1. KMeans color clustering
2. Triangulation
3. Seam insertion/deletion, for dynamically resizing images using a smart content-aware algorithm

## Examples ##
Original image:

![alt tag](http://i59.tinypic.com/azdks7.jpg)

1. triangulate image with ~20 triangle vertices on the width side of the image

    ```$ python3 improc.py -i images/sun.jpg -t 20 -o images/tri_20.jpg```
![alt tag](http://i58.tinypic.com/2zswtua.jpg)

2. reduce the image to its 10 representative colors

    ```$ python3 improc.py -i images/sun.jpg -k 10 -o images/tri_5.jpg```
![alt tag](http://i62.tinypic.com/2nv73bl.jpg)
  
3. intelligently increase the image width by 50 px

    ```$ python3 improc.py -i images/sun.jpg -wi 50 -o images/wi_50.jpg```
![alt tag](http://i61.tinypic.com/24o3h1z.jpg)
    
4. intelligently decrease the width by 100 px

    ```$ python3 improc.py -i images/sun.jpg -wi -100 -o images/wi_minus_100.jpg```
![alt tag](http://i61.tinypic.com/64e3dc.jpg)

# Usage
See the examples above, or print help:
```
$ python3 improc.py 
usage: improc.py [-h] -i IMAGE [-k KMEANS] [-t TRIANGULATE] [-s] [-o OUTPUT]
                 [-std STDOUT] [-wi WIDTH] [-he HEIGHT]

Image processing tools

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        file path to source image to process
  -k KMEANS, --kmeans KMEANS
                        Quantize image using K-Means into a given number of
                        clusters
  -t TRIANGULATE, --triangulate TRIANGULATE
                        Triangulate image with a given complexity
  -s, --show            Show image after processing
  -o OUTPUT, --output OUTPUT
                        Output transformed image to a file
  -std STDOUT, --stdout STDOUT
                        Output transformed image to standard out using a
                        specified format (jpg, png, etc...)
  -wi WIDTH, --width WIDTH
                        Resize image width by removing low energy seams
  -he HEIGHT, --height HEIGHT
                        Resize image height by removing low energy seams
```