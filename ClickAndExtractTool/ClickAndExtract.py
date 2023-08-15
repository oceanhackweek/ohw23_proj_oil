#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This script loop at every image and the corresponding mask files in the directories, 
# and allows to click in the segment and extract some statistical metrics
# of the segment in the original image.
#
# Authors: J.R. Garcia and M. P. Graziotto
# Ocean Hack Week - 2023
#


import cv2  # BGR, np.array
import csv
import os
import sys
import pandas as pd
import numpy as np
import glob              # glob file name list
import Functions as Fun
import Fun_Stat as FunStat
import importlib
importlib.reload(Fun)
importlib.reload(FunStat)

# For every image in dir_img there must be one in dir_mask (with same name)
dir_img   = './data/2-output_patches'
dir_mask  = './data/2-output_labels'

# Remember: OpenCV is BGR, PIL is RGB, mpl is RGB
scr_wid, scr_hei = Fun.getLargerMonitor()
  
def msg(text, msg_type="y"):
   ESC = "\x1B["
   if msg_type == "y":             # ok 
      print(ESC + "30;42m" + text) 
   elif msg_type == "i":           # info
      print(ESC + "97;44m" + text)   
   elif msg_type == "n":           # alert
      print(ESC + "97;41m" + text)   
   elif msg_type == "x":           # não-óleo
      print(ESC + "48;5;196m" + text)   
   else:
      print(ESC + "97;100m" + text) # grey bg 
      
   print(ESC + "0m")  # RESET ALL ATTRIBS


# Font definition
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 10
color = (255,0,255)
thickness = 2

# Callback function for mouse interation over the image, captures clicks and movements
# Think that when this method is executed, the window with 3 imgs is already been shown
def click_and_do(event, x, y, flags, param):
   # grab references to the global variables
   global sel_mask, img_work, img_show, indexes  # allows changing global var
   
   # if left mouse button was clicked ...
   if event == cv2.EVENT_LBUTTONDOWN:
      # User must click on img_label (the one in the middle) only
      if x < img.shape[1] or x > img.shape[1] * 2:
         return
      # x=1183; y=158
      sel_mask = img_show[y, x]
      print(f"CLICKED ON x={x}, y={y}, sel_mask={sel_mask}")
      img_work = img.copy()
      mask = cv2.inRange(img_label, sel_mask, sel_mask)
      indexes = np.where(mask != 0)  
      # len(indexes[0]), len(indexes[1]), img.shape, img.shape[0]*img.shape[1]
      img_work[indexes[0], indexes[1], :] = sel_mask
      img_show = cv2.hconcat([img, img_label, img_work])

      text = "Original"
      coordinates = (100, img_wid/2)
      #img_show = cv2.putText(img_show, "Original"   , (100                      , 10), font, fontScale, color, thickness, cv2.LINE_AA)
      #img_show = cv2.putText(img_show, "Mask"       , (img_wid + (img_wid/2)    , 10), font, fontScale, color, thickness, cv2.LINE_AA)
      #img_show = cv2.putText(img_show, "Masked  ori", ((img_wid*2) + (img_wid/2), 10), font, fontScale, color, thickness, cv2.LINE_AA)
      cv2.imshow("img_show", img_show) # Shows the new image over the old one
      #cv2.setWindowProperty("img_show", cv2.WND_PROP_TOPMOST, 0)
      #key = cv2.waitKey(0) & 0xFF
      #cv2.destroyAllWindows()

# Getting all fnames
fnames = sorted(glob.glob(dir_img + '/*.jpg'))
idx_img=0; fname_img = fnames[idx_img]
header = ['img_name', 'mask', 'mean', 'geometric_mean', 'harmonic_mean', 'median', 'mode', 'quantiles']
STATS_DF = pd.DataFrame(columns=header)
for idx_img, fname_img in enumerate(fnames):
   img_ori  = cv2.imread(fname_img)                       # ***** BGR FORMAT ******
   fname_label = fname_img.replace("patches", "labels").replace("jpg", "png")
   img_label  = cv2.imread(fname_label)                    # ***** BGR FORMAT ******
   img_ori_hei, img_ori_wid, _ = img_ori.shape
   head, tail = os.path.split(fname_img)
   
   # Statistics stored in a file with the same basename of the original image file name
   fname_stat = f"./3-stats/stat_{tail.replace('jpg', 'csv')}"

   print("---------------------------------------------------------------------------")
   print("* Img fname..: ", tail, ": ", sep="",)
   print("* Img ori dim: ", img_ori_wid, "x", img_ori_hei, ", asp_ratio: ", img_ori_hei / img_ori_wid, ", resizing, ", sep="", end="", flush=True) 
   
   # Resize the images to fit 3 of them, side by side
   img, fac_wid, fac_hei = Fun.resizeImg(img_ori, scr_wid, scr_hei)
   img_label, fac_wid, fac_hei = Fun.resizeImg(img_label, scr_wid, scr_hei)
   img_work = img.copy()
   img.shape, img_label.shape, img_work.shape
   assert img.shape == img_label.shape == img_work.shape

   print("---------------------------------------------------------------------------")
   img_hei, img_wid, _ = img.shape
   indexes = None

   # Creating a named placeholder for the image that will suffer mouse movements consequences
   # in order to assign a callback function to it before the creation of the window itself
   cv2.namedWindow("img_show")
   cv2.setMouseCallback("img_show", click_and_do)
   cv2.setWindowProperty("img_show", cv2.WND_PROP_TOPMOST, 1)

   print("\nControls:\nMouse down: select mask\n(N) Next Image \
          \n(X) Extract masked info\n(Q) Quit\n")

   # keep looping until the 'q' key is pressed
   key=""
   while True:
      # Concatenate 3 images horizontally
      img_show = cv2.hconcat([img, img_label, img_work])
      cv2.imshow("img_show", img_show)
      # Workaround to return the focus to the image window
      cv2.setWindowProperty("img_show", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
      key = cv2.waitKey(0) & 0xFF
      cv2.setWindowProperty("img_show", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL);

      if key == ord("x"):    # toggle crossed lines on/off
         print("Extracting statistics:")
      elif key == ord("n"):    # next image
         break
      elif key == ord("q"):    # quit
         break
      # Extract 
      if key == ord("x"):
         if indexes is not None:
            data = img[indexes[0], indexes[1], 0]  # all color dimension are equal, only one is needed
            stats = list((tail, 
                          ",".join(str(x) for x in sel_mask)) + 
                          FunStat.get_stats(data))
            #print(stats)
            STATS_DF.loc[len(STATS_DF)] = stats
            print(STATS_DF)
            #print(STATS_DF.drop_duplicates(inplace=True))
         else:
            print(" *** No segment selectd ***")
         if key == ord("o"):    # register and leave
            break

   # Leave image iteration
   if key == ord("q") or key == ord("t"):    # next image
      break
        
   # close all cv2 windows
   cv2.destroyAllWindows()    

