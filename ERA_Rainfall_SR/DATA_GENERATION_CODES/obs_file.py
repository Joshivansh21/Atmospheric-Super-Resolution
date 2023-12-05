import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Concatenate
from numpy import expand_dims
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as mtpltcm
from mpl_toolkits.mplot3d import Axes3D
from netCDF4 import Dataset
import cartopy.crs as ccrs
from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning) 
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)

import os 
import glob

#PATH_tcw = "data/twc_tcw.nc"
#PATH_tp="data/tp_tp.nc"
#PATH_temp="data/temp_t2m.nc"
#PATH_slp="data/slp_msl.nc"
#PATH_cpe="data/cpe_cape.nc"
PATH_obs="obsdata/"

#TCW=Dataset(PATH_tcw,more="r")
#TP=Dataset(PATH_tp,more="r")
#TEMP=Dataset(PATH_temp,more="r")
#SLP=Dataset(PATH_slp,more="r")
#CPE=Dataset(PATH_cpe,more="r")

files = glob.glob(os.path.join(PATH_obs, "*.nc"))
Highres= [Dataset(file,more="r") for file in files]
Dataobs=[[],[],[],[],[]]

for j in range(len(Highres)):
	if (Highres[j].variables["RAINFALL"].shape[0]==365):
		loop=range(152,274,1)
	else:
		loop=range(153,275,1)

	for i in loop:
		D=np.array(Highres[j].variables["RAINFALL"][i,:,:])
		if (np.amax(D)!=0):
            		D=D/np.amax(D)
		D=D[2:,3:-4]
		D[D<0]=-1.0
		npad=((0,1),(0,0))
		D=np.pad(D, pad_width=npad, mode='constant', constant_values=-1.0)
	
		Dataobs[0].append(D[0:48,15:63])
		Dataobs[1].append(D[40:88,0:48])
		Dataobs[2].append(D[40:88,40:88])
		Dataobs[3].append(D[50:98,80:128])
		Dataobs[4].append(D[80:128,20:68])

def obsfunc(region):
	region=tf.convert_to_tensor(region)
	region=tf.expand_dims(region, axis=-1)
	region=tf.cast(region,tf.float32)
	return region

northobs=obsfunc(Dataobs[4])
westobs=obsfunc(Dataobs[1])
centralobs=obsfunc(Dataobs[2])
eastobs=obsfunc(Dataobs[3])
southobs=obsfunc(Dataobs[0])
Highres_data=Concatenate(axis=0)([northobs, westobs, centralobs, eastobs, southobs])
print(Highres_data.shape)
print("Highres_data done")
np.save("HPCTEST", Highres_data )

#long1=np.array(Highres[0].variables["LONGITUDE"][3:-4])
#lat1=np.array(Highres[0].variables["LATITUDE"][2:])
#lat1=np.append(lat1,38.75)
#(ln5, lt5) = (long1[15:63], lat1[0:48])
#(ln2, lt2) = (long1[0:48], lat1[40:88])
#(ln3, lt3) = (long1[40:88], lat1[40:88])
#(ln4, lt4) = (long1[80:128], lat1[50:98])
#(ln1, lt1)=(long1[20:68], lat1[80:128])
#OUT=Highres_data[4880]
#OUT=ma.masked_less(OUT,0)
#OUT=np.reshape(OUT, (48, 48))
#OUTFULL=np.array(Highres[0].variables["RAINFALL"][6,:,:])
#if (np.amax(OUTFULL)!=0):
#	OUTFULL=OUTFULL/np.amax(OUTFULL)
#OUTFULL[OUTFULL<0]=-1.0
#OUTFULL=OUTFULL[2:,3:-4]
#npad=((0,1),(0,0))
#OUTFULL=np.pad(OUTFULL, pad_width=npad, mode='constant', constant_values=-1.0)
#OUTFULL=ma.masked_less(OUTFULL,0)


#plt.style.use("seaborn-v0_8-bright")
#figure = plt.figure(figsize=(10,10))
#axis_func = plt.axes(projection=ccrs.PlateCarree())
#axis_func.coastlines(resolution="10m",linewidth=1)
#axis_func.gridlines(linestyle='--',color='black',linewidth=2)
#N=99
#plt.contourf(ln2, lt2,OUT , N, transform=ccrs.PlateCarree(), cmap='terrain')
#color_bar_func = plt.colorbar(ax=axis_func, orientation="vertical", pad=0.05, aspect=16, shrink=.8)
#color_bar_func.ax.tick_params(labelsize=8)
#plt.tight_layout()
#plt.savefig('west6.png')
#plt.show()
