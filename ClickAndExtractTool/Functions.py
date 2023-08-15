#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:11:02 2022

@author: josegarcia
"""

import math
import cv2
import re
import os
import numpy as np
import pandas as pd
import glob              # glob file name list
from screeninfo import get_monitors
import matplotlib.pyplot as plt
from sys import platform
import Fun_Color as fcol

if platform == "linux" or platform == "linux2":
   plat = "linux"
elif platform == "darwin":
   plat = "mac" # OS X
elif platform == "win32":
   plat = "windows"
   
def pdConcat(DF_RECT, RES, col_names):
    if len(RES.shape) == 3:
        X = pd.concat([
            DF_RECT, 
            pd.Series(RES[:,:, 2].ravel(), dtype="int16"),           # R
            pd.Series(RES[:,:, 1].ravel(), dtype="int16"),           # G
            pd.Series(RES[:,:, 0].ravel(), dtype="int16")], axis=1)  # B
    else:
        X = pd.concat([
            DF_RECT, 
            pd.Series(RES.ravel(), dtype="float16")], axis=1)  # GREY SCALE
    X.columns = list(DF_RECT) + col_names
    if X.isnull().values.any():
        X=None
    return(X)


def exploreRect(rect, img_ori, 
                expl_type=["RGB", "MAX_RGB", "BOOST_RGB", "BLUR", "POWER", "GREY_RGB", "STD", "AVG", "MAX_DIFF", 
                           "GREY_HSV", "GREY_LAB", "GREY_YCrCb", "GREY_CMY", "GREY_CV", "GREY_DIV"],
                quiet=False):
    # GUARDA A EXPLORAÇÃO DE CADA RECT, BIND DE COLUNAS ATÉ O FINAL DO LOOP
    DF_RECT = pd.DataFrame(index=None)  
    
    # First Y then X
    rect3D = img_ori[int(rect.Y1):int(rect.Y2+1), int(rect.X1):int(rect.X2+1)]
    #plt.imshow(rect3D)
    if "RGB" in expl_type:
       RES = rect3D.copy()
       vCharac = ["R", "G", "B"]
       if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
       if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
       DF_RECT = pdConcat(DF_RECT, RES, vCharac)
    
    # --------------------------------------------------------------------------
    # Colored images
    # --------------------------------------------------------------------------
    # manter maior valor entre BGR e zera os outros
    if "MAX_RGB" in expl_type:
       RES = fcol.max_rgb(rect3D)        
       vCharac = ["MAX_R", "MAX_G", "MAX_B"]
       if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
       if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
       DF_RECT = pdConcat(DF_RECT, RES, vCharac)
    
    # maximiza o maior valor entre BGR (põe 255)
    if "BOOST_RGB" in expl_type:
       RES = fcol.boost_rgb(rect3D)     
       vCharac = ["BST_R", "BST_G", "BST_B"]
       if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
       if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
       DF_RECT = pdConcat(DF_RECT, RES, vCharac)
    
    # Blured (smoothed)
    if "BLUR" in expl_type:
       RES = cv2.GaussianBlur(rect3D, (5, 5), cv2.BORDER_DEFAULT)
       vCharac = ["BLUR_GAUS_R", "BLUR_GAUS_G", "BLUR_GAUS_B"]
       if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
       if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
       DF_RECT = pdConcat(DF_RECT, RES, vCharac)
    
       RES = cv2.bilateralFilter(rect3D, 5, 75, 75)
       vCharac = ["BLUR_BILAT_R", "BLUR_BILAT_G", "BLUR_BILAT_B"]
       if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
       if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
       DF_RECT = pdConcat(DF_RECT, RES, vCharac)
       
       RES = cv2.medianBlur(rect3D, 5)
       vCharac = ["BLUR_MED_R", "BLUR_MED_G", "BLUR_MED_B"]
       if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
       if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
       DF_RECT = pdConcat(DF_RECT, RES, vCharac)

    
    # Colored images - BGR power
    if "POWER" in expl_type:
       RES = fcol.getImgChannelPower(rect3D, "R")
       vCharac = ["R_POW"]
       if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
       if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
       DF_RECT = pdConcat(DF_RECT, RES, vCharac)
       
       RES = fcol.getImgChannelPower(rect3D, "G")
       vCharac = ["G_POW"]
       if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
       if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
       DF_RECT = pdConcat(DF_RECT, RES, vCharac)
       
       RES = fcol.getImgChannelPower(rect3D, "B")
       vCharac = ["B_POW"]
       if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
       if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
       DF_RECT = pdConcat(DF_RECT, RES, vCharac)
    
 
    #pal=["black", "grey", "white", "blue", "darkblue"]
    #lut = fcol.rgb2LkUpTb(fcol.getColors(pal, n=256))
    #B_pow_cm = cv2.LUT(B_pow, lut)
    #B_pow_cm = cv2.applyColorMap(B_pow, cv2.COLORMAP_JET)
    #cuts = np.linspace(start=R_pow.min(), stop=R_pow.max(), num=5)
    #cmap = colors.ListedColormap(pal)
    #plt.imshow(R_pow, cmap=cmap, vmin=R_pow.min(), vmax=R_pow.max())
    #plt.clim([min(cuts), max(cuts)])
 
    if "GREY_RGB" in expl_type:
        # BGR in grey scales
        G1, G2, G3 = cv2.split(rect3D)           # Cria tons de cinza para os valores BGR isoladamente
        RES = G1
        vCharac = ["GREY_R"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
        RES = G2
        vCharac = ["GREY_G"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
        RES = G3
        vCharac = ["GREY_B"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
    
    if "STD" in expl_type:
        # Custom grey scales
        RES = fcol.greyScaleBy(rect3D, by="STD")         # std de cada tupla BGR
        vCharac = ["STD"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
 
    if "AVG" in expl_type:
        RES = fcol.greyScaleBy(rect3D, by="AVG")         # média de cada tupla BGR (grey scale)
        vCharac = ["AVG"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
    
    if "MAX_DIFF" in expl_type:
        RES  = fcol.greyScaleBy(rect3D, "MAX_DIFF")
        vCharac = ["MAX_DIFF"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
    
    # Color space HSV in grey scales
    if "GREY_HSV" in expl_type:
        cs = cv2.cvtColor(rect3D, cv2.COLOR_BGR2HSV)
        G1, G2, G3 = cv2.split(cs)
        RES = G1
        vCharac = ["GREY_H"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
        RES = G2
        vCharac = ["GREY_S"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
        RES = G3
        vCharac = ["GREY_V"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
    
    # Color space LAB in grey scale
    if "GREY_LAB" in expl_type:
        cs = cv2.cvtColor(rect3D, cv2.COLOR_BGR2LAB)
        G1, G2, G3 = cv2.split(cs)
        RES = G1
        vCharac = ["GREY_LIGHT"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
        RES = G2
        vCharac = ["GREY_LA"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
        RES = G3
        vCharac = ["GREY_LB"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
 
    # Color space BGR2YCrCb in grey scale
    if "GREY_YCrCb" in expl_type:
        cs = cv2.cvtColor(rect3D, cv2.COLOR_BGR2YCrCb)
        G1, G2, G3 = cv2.split(cs)
        RES = G1
        vCharac = ["GREY_LUMI"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
        RES = G2
        vCharac = ["GREY_Cr"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
        RES = G3
        vCharac = ["GREY_Cb"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
    
    # Color space CMY (Cyam, Magenta, Yellow) in grey space
    if "GREY_CMY" in expl_type:
        G1, G2, G3 = fcol.rgb_to_cmy(rect3D)
        RES = G1
        vCharac = ["GREY_C"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
        RES = G2
        vCharac = ["GREY_M"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
        RES = G3
        vCharac = ["GREY_Y"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
    
    # More grey scales
    if "GREY_CV" in expl_type:
        RES   = cv2.cvtColor(rect3D, cv2.COLOR_BGR2GRAY) 
        vCharac = ["BGR2GRAY"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
    
    if "GREY_DIV" in expl_type:
        RES = fcol.greyScaleBy(rect3D, "HUMAN")  # OR HDTV
        vCharac = ["HUMAN"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
    
        RES = fcol.greyScaleBy(rect3D, "HDR")
        vCharac = ["HDR"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
            
        # More grey scales
        RES = fcol.greyScaleBy(rect3D, "LIGHTNESS")
        vCharac = ["LIGHTNESS"]
        if np.isnan(np.sum(RES)) != 0: print(f"Error when concatenating {vCharac}"); return(None)
        if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
        DF_RECT = pdConcat(DF_RECT, RES, vCharac)
        
        RES = fcol.greyScaleBy(rect3D, "LUMINANCE2")
        vCharac = ["LUMINANCE"]
        
    if not quiet: print(" " * line_len, sep='', end="\r", flush=True); print(f"{pmsg} {vCharac}", sep="", end="\r", flush=True)
    #DF_RECT = pdConcat(DF_RECT, RES, vCharac)
    return(DF_RECT)


# Verifica se o arquivo existe em algum dos diretórios e retorna seu path completo
#fname = tail
def fileExist(fname):
   fname = fname.replace(".csv", "") 
   # A imagem pode estar num destes diretórios abaixo
   fname_full = f"{DIR.img}{fname}"
   if not os.path.exists(f"{fname_full}"):
      fname_full = f"{DIR.selected}{fname}"
      if not os.path.exists(f"{fname_full}"):
         fname_full = f"{DIR.annotated}train/{fname}"
         if not os.path.exists(f"{fname_full}"):
            fname_full = f"{DIR.annotated}test/{fname}"
            if not os.path.exists(f"{fname_full}"):
               fname_full = None
   return(fname_full)

# -----------------------------------------------------------------------------
# Lê todos os arquivos CSV do diretório e carrega num DF Pandas
# -----------------------------------------------------------------------------
#dsRGB = pd.concat(map(lambda file: pd.read_csv(file, sep=";", header=None), 
#                      glob.glob(os.path.join(DIR.clipped, "*.csv"))), axis=0)
def getClips(fpath, remove_csv=True):
    fnames_clipped = [os.path.basename(x) for x in sorted(glob.glob(fpath + '*.csv'))]
    ds_clipped = pd.DataFrame()
    str_change = "" if remove_csv else ".csv" 
    for f in fnames_clipped:
        DS = pd.read_csv(f"{fpath}{f}", sep=";", header=None)
        ds_clipped = pd.concat([ds_clipped, 
                               pd.concat([pd.Series([f.replace(".csv", str_change)] * len(DS)), DS], axis=1)])
    ds_clipped.columns = ["FNAME", "FUNDO", "X1", "X2", "Y1", "Y2"]    
    ds_clipped.reset_index(drop=True, inplace=True)
    return(ds_clipped)

# Inclui plots em subplots já criados
def plotFeed(img, title, xaxis=False, yaxis=False):
    plt.imshow(img)
    plt.title(title)
    if not xaxis: 
       plt.xticks([])
    if not yaxis: 
       plt.yticks([])
    return(plt)

# Captura a dimensão do maior monitor quando mode=True, otherwise, menor
def getLargerMonitor(larger=True, shrink_factor=0):
    mon_widths = []
    mon_heights = []
    mon_dims = []
    
    try:   # O SO pode não aceitar esta função
        monitors = get_monitors()
        print("* Monitors detected: ", end="", flush=True)
        for monitor in monitors:
            mon_widths += [monitor.width]
            mon_heights += [monitor.height]
            mon_dims += [monitor.height * monitor.width]
            print(monitor.width, "x", monitor.height, ", ", sep="", end="")
            
        if max:
           larg_mon = mon_dims.index(max(mon_dims))
        else:
           larg_mon = mon_dims.index(min(mon_dims))
        
        ret_wid = int(monitors[larg_mon].width * (1-shrink_factor/100))
        ret_hei = int(monitors[larg_mon].height * (1-shrink_factor/100))
    except:  # então captura da linha de comando do SO
        if plat == "mac":
           os.system("system_profiler SPDisplaysDataType | grep 'Resolution:' | head -1 > mon_resolution.txt")
           pattern = re.compile("[0-9]* *x *[0-9]*")
           idx_wid = 0; idx_hei = 1; split_char="x"
        elif plat == "linux":
           os.system("xdpyinfo | grep 'dimensions:' | head -1 > mon_resolution.txt")
           pattern = re.compile("[0-9]* *x *[0-9]*")
           idx_wid = 0; idx_hei = 1; split_char="x"
        else: #plat == "windows":
           os.system("wmic desktopmonitor get screenheight, screenwidth > mon_resolution.txt")
           os.system("more +1 mon_resolution.txt > mon_resolution.txt")
           pattern = re.compile("[0-9]* *[0-9]*")
           idx_wid = 1; idx_hei = 0; split_char=" "
            
        with open('mon_resolution.txt', 'r') as f:
           res_text = f.read()
           #res_text = "5120 2880"
        
        res = pattern.search(res_text).group(0)
        ret_wid, ret_hei = (int(res.split(split_char)[idx_wid]), int(res.split(split_char)[idx_hei]))
        ret_wid = int(ret_wid * (1-shrink_factor/100))
        ret_hei = int(ret_hei * (1-shrink_factor/100))
       
    print("chosen: ", ret_wid, "x", ret_hei, sep="", flush=True)
    return(ret_wid, ret_hei)

    #print(str(width) + 'x' + str(height))



# Check if resizing is needed, must be known then
# img precisa ser gerada com cv2.imread()
# O aspect ratio da imagem deve ser mantido e a escala a ser adotada depende 
# do quanto a imagem exrapola o maior monitor em larg ou alt
#img=img_ori; scr_wid, scr_hei = getLargerMonitor(); showw_mode="M"
def resizeImg(img, scr_wid, scr_hei, show_mode="F", quiet=False):
    # cv2 = np.ndarray = shape, PIL = JpegImage = size
    img_hei, img_wid, _ = img.shape
    asp_rat = img_hei / img_wid  # aspect_ratio da img deve ser mantido

    # Quando manchas, divide width por e height por 4 (32 imagens da mancha)    
    if show_mode == "M":
       scr_wid /= 8
       scr_hei /= 4
    #print(img_wid, img_hei)
    # Se a imagem não couber no maior monitor em alguma das duas dimensões
    if img_wid > scr_wid/3 or img_hei > scr_hei:
        # Quão perto as dimensões da imagem são das dimensões do maior monitor?
        # Quanto menor mais diferente (longe) é, então devemos usar o fator menor
        # e adaptar o outro de acordo com o aspect ratio
        fac_wid = scr_wid/3 / img_wid
        fac_hei = scr_hei / img_hei
        if fac_wid < fac_hei:
            new_img_wid = math.floor(img_wid * fac_wid)
            new_img_hei = math.floor(img_hei * fac_wid)
        else:
            new_img_hei = math.floor(img_hei * fac_hei)
            new_img_wid = math.floor(img_wid * fac_hei)
            
        # Novo aspect ratio deve ser gual ao original
        asp_rat_new = new_img_hei / new_img_wid
            
        # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4
        # resize() function 
        fx = img_wid / new_img_wid
        fy = img_hei / new_img_hei
        
        if not quiet:
           print("* Img new dim: ", new_img_wid, "x", new_img_hei, ", asp_ratio: ", asp_rat_new, 
                 ", fx: ", fx, ", fy: ", fy, sep="", flush=True)
        
        return(cv2.resize(img, (new_img_wid, new_img_hei), interpolation = cv2.INTER_AREA), fx, fy)
    else:
        if not quiet:
           print("* No resizing needed (the 3 images fit screen width)", sep="", flush=True)
        # Se não precisa de resize fx e fy = 1
        return(img, 1, 1)


# Dá um zoom na parte cropada da image de acordo com facscale pois é muito pequena
# img precisa ser gerada com cv2.imread()
# fac_scale default=200%
def zoomCrop(img, fac_scale=200):
    img_wid = img.shape[0] if type(img) == np.ndarray else img.size[0]
    img_hei = img.shape[0] if type(img) == np.ndarray else img.size[1]
    new_img_wid = math.floor(img_wid * (1 + fac_scale / 100))
    new_img_hei = math.floor(img_hei * (1 + fac_scale / 100))
    return(cv2.resize(img, (new_img_wid, new_img_hei), interpolation = cv2.INTER_AREA))
    