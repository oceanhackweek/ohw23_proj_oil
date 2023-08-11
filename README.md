# Oil spill Monitoring: Segmentation of Satellite Imagery

## One-line Description
This is a Python notebook which crops a satellite image of a possible oil slick and tries to determine - using simple image statistics - if it's an actual oil spill or a look-alike; this is the first stage of a greater software pipeline to automate identifying oil spills from satellite imagery. 

## Collaborators
Amanda D., Maria G., José G., Jaelyn Bos, Anand Sekar

## Background/ Goals
The main idea of this project is to automate the process of identifying a possible oil slick in a satellite image, which involves these main steps: 
 1. Loading: Reading the satellite image file(s)
 2. Cropping: efficient pre-processing, such as a simple statistical analysis (e.g. histogram), to identify possible oiled areas and crop them into patches
 3. Segmentation: segment the oiled areas from the patches (future)
 4. Presentation: visualize the segments: make a list, save metadata (future)

## Datasets
(TODO: elaborate further) These images were taken from the Sentinel-1 satellite (?). 

## References/ Literature Review


