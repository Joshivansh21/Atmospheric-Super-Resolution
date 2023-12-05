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

filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)
PATH_files="validation_data\high_res"
files = glob.glob(os.path.join(PATH_files, "*.nc"))
Scale_data=[Dataset(file, more="r") for file in files]
INPUT=np.load("3Fieldtest.npy")
OUTPUT=np.load("3Fieldhighrestest.npy")
model = load_model("pretrain_2_3field\Pretraining_2_3Channels29.0.keras", compile=False)
IMAGE=998
scale_factor=[]
for i in range(len(Scale_data)):
	if (Scale_data[i].variables["RAINFALL"].shape[0]==365):
		loop=range(152,274,1)
	else:
		loop=range(153,275,1)
	for j in loop:
		Dp=np.array(Scale_data[i].variables["RAINFALL"][j,:,:])
		Dp[Dp<=0]=0
		scale_factor.append(np.amax(Dp))
scale_input=5*scale_factor




gen=INPUT[IMAGE]
imd=OUTPUT[IMAGE]
imd=np.ma.masked_less(imd,0)
imd=np.reshape(imd, (48,48))
#imd=imd*scale_input[IMAGE]
rain_gen=gen[:,:,:,1]
sum=np.zeros((12,12))
for i in range(rain_gen.shape[0]):
  sum=sum+rain_gen[i,:,:]
sum=np.ma.masked_less(sum,0)
#sum=sum*scale_input[IMAGE]

mask=ma.getmask(Scale_data[0].variables["RAINFALL"][0,2:,3:-4])
gen=np.reshape(gen, (1, 24, 12, 12, 3))
OUT = model.predict(gen)
print(OUT.shape)
OUT=np.reshape(OUT,(48,48))
for j in range(mask.shape[0]):
	for k in range(mask.shape[1]):
		if mask[j][k]== True:
			Dp[j][k]=-1
OUT[OUT.shape[0]-1][:]=-1
OUT=np.ma.masked_less(OUT,-0.5)
print(np.amin(OUT))
#OUT=OUT-np.amin(OUT)
OUT[OUT<0]=0
#OUT=OUT*scale_input[IMAGE]






imdlat=np.array(Scale_data[0].variables["LATITUDE"][2:])
imdlong=np.array(Scale_data[0].variables["LONGITUDE"][3:-4])
imdlat=np.append(imdlat,38.75)

(iln5, ilt5) = (imdlong[20:68], imdlat[0:48])
(iln2, ilt2) = (imdlong[0:48], imdlat[40:88])
(iln3, ilt3) = (imdlong[40:88], imdlat[40:88])
(iln4, ilt4) = (imdlong[80:128], imdlat[50:98])
(iln1, ilt1)= (imdlong[15:63], imdlat[80:128])

(loiln5, loilt5) = (iln5[::4], ilt5[::4])
(loiln2, loilt2) = (iln2[::4], ilt2[::4])
(loiln3, loilt3) = (iln3[::4], ilt3[::4])
(loiln4, loilt4) = (iln4[::4], ilt4[::4])
(loiln1, loilt1)= (iln1[::4], ilt1[::4])

if (IMAGE//244==0):
	(lat, long)=(iln1, ilt1)
	(ilat, ilong)=(loiln1, loilt1) 
elif (IMAGE//244==1):
	(lat, long)=(iln2, ilt2)
	(ilat, ilong)=(loiln2, loilt2)
elif (IMAGE//244==2):
	(lat, long)=(iln3, ilt3)
	(ilat, ilong)=(loiln3, loilt3)
elif(IMAGE//244==3):
	(lat, long)=(iln4, ilt4)
	(ilat, ilong)=(loiln4, loilt4)
else:
	(lat, long)=(iln5, ilt5)
	(ilat, ilong)=(loiln5, loilt5)

#print(lat,long, ilat, ilong)

#plt.style.use("seaborn-v0_8-bright")
#figure = plt.figure(figsize=(10,10))
#axis_func = plt.axes(projection=ccrs.PlateCarree())
#axis_func.coastlines(resolution="10m",linewidth=1)
#axis_func.gridlines(linestyle='--',color='black',linewidth=2)
#N=99
#plt.contourf(ilong, ilat, sum, N, transform=ccrs.PlateCarree(), cmap='terrain')
#color_bar_func = plt.colorbar(ax=axis_func, orientation="vertical", pad=0.05, aspect=16, shrink=.8)
#color_bar_func.ax.tick_params(labelsize=12)
#plt.tight_layout()
##plt.savefig('dp4.png')
#plt.show()
plotmax=max(np.amax(imd),np.amax(OUT))
level=np.linspace(0, plotmax, 100)

plt.style.use("seaborn-v0_8-bright")
figure = plt.figure(figsize=(10,10))
axis_func = plt.axes(projection=ccrs.PlateCarree())
axis_func.coastlines(resolution="10m",linewidth=1)
axis_func.gridlines(linestyle='--',color='black',linewidth=2)
N=99
plt.contourf(lat, long,  OUT, N, transform=ccrs.PlateCarree(), cmap='terrain', levels=level)
color_bar_func = plt.colorbar(ax=axis_func, orientation="vertical", pad=0.05, aspect=16, shrink=.8)
color_bar_func.ax.tick_params(labelsize=12)
plt.tight_layout()
#plt.savefig('dp4.png')
#plt.show()

plt.style.use("seaborn-v0_8-bright")
figure = plt.figure(figsize=(10,10))
axis_func = plt.axes(projection=ccrs.PlateCarree())
axis_func.coastlines(resolution="10m",linewidth=1)
axis_func.gridlines(linestyle='--',color='black',linewidth=2)
N=99
plt.contourf(lat, long , imd, N, transform=ccrs.PlateCarree(), cmap='terrain', levels=level)
color_bar_func = plt.colorbar(ax=axis_func, orientation="vertical", pad=0.05, aspect=16, shrink=.8)
color_bar_func.ax.tick_params(labelsize=12)
plt.tight_layout()
##plt.savefig('dp4.png')
plt.show()
