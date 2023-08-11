# Oil spill Monitoring: Segmentation of Satellite Imagery

## One-line Description
This is a Python notebook which crops a SAR (Synthetic Aperture Radar) image of a possible oil slick and tries to determine - using simple image statistics - if it's an actual oil spill or a look-alike. 

## Collaborators
Amanda D., Maria G., José G., Jaelyn Bos, Anand Sekar

## Background
This OHW project is the first stage of a greater software project to automate identifying oil spills from satellite imagery called Project SisMOM - Oil Monitoring System at Sea. 

## Goals
The main idea of this project is to automate the process of identifying a possible oil slick in a satellite image, which involves these main steps: 
 1. Loading: Reading the satellite image file(s)
 2. Cropping: efficient pre-processing, such as a simple statistical analysis (e.g. histogram), to identify possible oiled areas and crop them into patches
 3. Segmentation: segment the oiled areas from the patches (future)
 4. Presentation: visualize the segments: make a list, save metadata (future)

## Datasets
The images used were taken from a [SAR-2000](https://space.oscar.wmo.int/instruments/view/sar_2000) imaging sensor on the second [COSMO-SkyMed](https://earth.esa.int/eogateway/missions/cosmo-skymed) satellite called [CSKS2](https://space.oscar.wmo.int/satellites/view/csk_2). 

## References/ Literature Review
### More relevant:
 1. [Sensors, Features, and Machine Learning for Oil Spill Detection and Monitoring: A Review](https://www.mdpi.com/2072-4292/12/20/3338): contains statistics for pre-processing
 2. [An improved semantic segmentation model based on SVM for marine oil spill detection using SAR image](https://www.sciencedirect.com/science/article/pii/S0025326X23004137): From this July! The introduction to this paper cites some of the other papers in this list and summarizes previous efforts broadly.
 3. [Ocean oil spill detection from SAR images based on multi-channel deep learning semantic segmentation](https://www.sciencedirect.com/science/article/pii/S0025326X23000826): From this March! The introduction does a more thorough job of summarizing efforts, focusing on deep learning. 
### Other papers: 
 * [Oil Spill Identification from Satellite Images Using Deep Neural Networks](https://www.mdpi.com/2072-4292/11/15/1762)
   * [Oil Spill Detection Dataset](https://m4d.iti.gr/oil-spill-detection-dataset/)
 * [Oil Spill Classification Kaggle dataset](https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection/discussion)
 * [(Book) Automatic Detection Algorithms of Oil Spill in Radar Images](https://www.taylorfrancis.com/books/mono/10.1201/9780429052965/automatic-detection-algorithms-oil-spill-radar-images-maged-marghany)
 * [Oil spill detection by imaging radars: Challenges and pitfalls](https://www.sciencedirect.com/science/article/pii/S0034425717304145): focuses on the difficult parts of this effort
 * [Improving the RST-OIL Algorithm for Oil Spill Detection under Severe Sun Glint Conditions](https://www.mdpi.com/2072-4292/11/23/2762)
 * [A novel deep learning instance segmentation model for automated marine oil spill detection](https://www.sciencedirect.com/science/article/pii/S0924271620301982): uses mask-rcnn
 * [Marine oil spill detection using Synthetic Aperture Radar over Indian Ocean](https://www.sciencedirect.com/science/article/pii/S0025326X20310390)
 * [A novel deep learning method for marine oil spill detection from satellite synthetic aperture radar imagery](https://www.sciencedirect.com/science/article/pii/S0025326X22003484): "A large data set consisting of 15,774 labeled oil spill samples derived from 1786C-band Sentinel-1 and RADARSAT-2 vertical polarization SAR images is used to train, validate and test the Faster R-CNN model.”
 * [A Deep Convolutional Neural Network for Oil Spill Detection from Spaceborne SAR Images](https://www.mdpi.com/2072-4292/12/6/1015#)
 * [(META/ Literature Review) Oil Spill Detection and Mapping: A 50-Year Bibliometric Analysis](https://www.mdpi.com/2072-4292/12/21/3647)
   * [A really cool graphical abstract](https://pub.mdpi-res.com/remotesensing/remotesensing-12-03647/article_deploy/html/images/remotesensing-12-03647-ag.png?1604912763)
 * [Oil Spill Detection Based on Deep Convolutional Neural Networks Using Polarimetric Scattering Information From Sentinel-1 SAR Images](https://ieeexplore.ieee.org/document/9606718)
   * [Code](https://github.com/RS-xjg/oil-spill-detection)
 * [Feature Merged Network for Oil Spill Detection Using SAR Images](https://www.mdpi.com/2072-4292/13/16/3174)
 * [SAR Oil Spill Detection System through Random Forest Classifiers](https://www.mdpi.com/2072-4292/13/11/2044)
 * [Oil Spill Detection with Multiscale Conditional Adversarial Networks with Small-Data Training](https://www.mdpi.com/2072-4292/13/12/2378)
 * [Oil spill detection based on texture analysis: how does feature importance matter in classification?](https://www.tandfonline.com/doi/full/10.1080/01431161.2022.2106163)
 * [Oil Spill Detection Based on Multiscale Multidimensional Residual CNN for Optical Remote Sensing Imagery](https://ieeexplore.ieee.org/document/9591296)
 * [Decision Fusion of Deep Learning and Shallow Learning for Marine Oil Spill Detection](https://www.mdpi.com/2072-4292/14/3/666)
