# Transmission Electron Microscopy Image Segmenter (TEMIS)

TEMIS is an AI-driven image segmentation tool to segment helium (He) bubble regions in Transmission Electron Microscopy (TEM) images, 
used to understand the correlation between He ion implantation condition and the growth of He bubbles at different temperatures. 
To analyze the bubbles in TEM images, a labor-intensive manual annotation method using ImageJ was used previously, which took more than one hour to annotate bubble regions per image. 
TEMIS uses YOLOv8 model trained on He bubble regions labeled TEM images, automatically detecting and segmenting bubble regions on regular laptops without specific hardware/GPU requirements.  
TEMIS provides a PyQT-based user interface to interactively select and segment regions, with simple built-in tools for manual editing. 
I developed this program as a research intern and an external student collaborator at Lawrence Livermore National Laboratory for a materials science research project. 




## Prerequisites for Running Program

Python 3  
Pytorch  
OpenCV 4 or higher  
PyQT 5 or higher  
Astropy  
Pandas  

To install the program, run this command in this directory after cloning: conda env create -f environment.yml 


## Authors
Ian Kim  ik9davis@gmail.com  

## Version

Version 0.7 (beta)

## License

SPYDER is distributed under the terms of the MIT license. All new contributions must be made under this license.  
