import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import expand_dims
import numpy.ma as ma
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as mtpltcm
from mpl_toolkits.mplot3d import Axes3D
import random
import folium
from folium.plugins import HeatMap, HeatMapWithTime
from folium import plugins
from netCDF4 import Dataset
import cartopy.crs as ccrs
from tensorflow import keras
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Add
from keras.models import load_model
import numpy.ma as ma
from warnings import filterwarnings
import glob
import os
np.set_printoptions(threshold=np.inf)
IMAGE=16787
ERAIMAGE=456
path="IMD_DATA_MAX/Maxtemp_MaxT_1981.nc"
IMD=np.load("Inputfiletemp.npy")
ERA=np.load("Highresfiletemp.npy")
path2="ERA_DATA/temp_hourly_1981_1993.nc"
NETCDF_DATA=Dataset(path, more="r")
NETCDF2=Dataset(path2, more="r")


Highres=ERA[ERAIMAGE]
#print(Highres.type)
Highres=np.ma.masked_greater(Highres,50)
Highres=np.reshape(Highres, (48,48))
eralat=np.array(NETCDF2.variables["latitude"][:])
eralong=np.array(NETCDF2.variables["longitude"][:])
eralat=np.array(np.flip(eralat, axis=0))

(ln5, lt5) = (eralong[20:68], eralat[0:48])
(ln2, lt2) = (eralong[0:48], eralat[40:88])
(ln3, lt3) = (eralong[40:88], eralat[40:88])
(ln4, lt4) = (eralong[80:128], eralat[48:96])
(ln1, lt1) = (eralong[16:64], eralat[80:128])

if (ERAIMAGE//100==0):
	(elat, elong)=(ln1, lt1)
	#(ilat, ilong)=(loiln1, loilt1) 
elif (ERAIMAGE//100==1):
	(elat, elong)=(ln2, lt2)
	#(ilat, ilong)=(loiln2, loilt2)
elif (ERAIMAGE//100==2):
	(elat, elong)=(ln3, lt3)
	#(ilat, ilong)=(loiln3, loilt3)
elif(ERAIMAGE//100==3):
	(elat, elong)=(ln4, lt4)
	#(ilat, ilong)=(loiln4, loilt4)
else:
	(elat, elong)=(ln5, lt5)
	#(ilat, ilong)=(loiln5, loilt5)


temp= IMD[IMAGE]
temp=np.ma.masked_greater(temp,50)
temp=np.reshape(temp, (12,12))
imdlat=np.array(NETCDF_DATA.variables["latitude"][:])
imdlong=np.array(NETCDF_DATA.variables["longitude"][:])
imdlat=np.append(imdlat,38.5)
imdlong=np.append(imdlong,98.5)

(iln5, ilt5) = (imdlong[5:17], imdlat[0:12])
(iln2, ilt2) = (imdlong[0:12], imdlat[10:22])
(iln3, ilt3) = (imdlong[10:22], imdlat[10:22])
(iln4, ilt4) = (imdlong[20:32], imdlat[12:24])
(iln1, ilt1) = (imdlong[4:16], imdlat[20:32])

if (IMAGE//14244==0):
	(lat, long)=(iln1, ilt1)
	#(ilat, ilong)=(loiln1, loilt1) 
elif (IMAGE//14244==1):
	(lat, long)=(iln2, ilt2)
	#(ilat, ilong)=(loiln2, loilt2)
elif (IMAGE//14244==2):
	(lat, long)=(iln3, ilt3)
	#(ilat, ilong)=(loiln3, loilt3)
elif(IMAGE//14244==3):
	(lat, long)=(iln4, ilt4)
	#(ilat, ilong)=(loiln4, loilt4)
else:
	(lat, long)=(iln5, ilt5)
	#(ilat, ilong)=(loiln5, loilt5)

plt.style.use("seaborn-v0_8-bright")
figure = plt.figure(figsize=(10,10))
axis_func = plt.axes(projection=ccrs.PlateCarree())
axis_func.coastlines(resolution="10m",linewidth=1)
axis_func.gridlines(linestyle='--',color='black',linewidth=2)
N=99
plt.contourf(lat, long, temp, N, transform=ccrs.PlateCarree(), cmap='terrain')
color_bar_func = plt.colorbar(ax=axis_func, orientation="vertical", pad=0.05, aspect=16, shrink=.8)
color_bar_func.ax.tick_params(labelsize=12)
plt.tight_layout()
#plt.savefig('dp4.png')
plt.show()

plt.style.use("seaborn-v0_8-bright")
figure = plt.figure(figsize=(10,10))
axis_func = plt.axes(projection=ccrs.PlateCarree())
axis_func.coastlines(resolution="10m",linewidth=1)
axis_func.gridlines(linestyle='--',color='black',linewidth=2)
N=99
plt.contourf(elat, elong, Highres, N, transform=ccrs.PlateCarree(), cmap='terrain')
color_bar_func = plt.colorbar(ax=axis_func, orientation="vertical", pad=0.05, aspect=16, shrink=.8)
color_bar_func.ax.tick_params(labelsize=12)
plt.tight_layout()
#plt.savefig('dp4.png')
plt.show()