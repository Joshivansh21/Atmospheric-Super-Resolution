import numpy as np
import os 
import glob
import tensorflow as tf
from tensorflow import keras
from keras.layers import Concatenate
from numpy import expand_dims
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as mtpltcm
from mpl_toolkits.mplot3d import Axes3D
import random
from netCDF4 import Dataset
import cartopy.crs as ccrs
np.set_printoptions(threshold=np.inf)
from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning) 
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)



PATH_tcw = "data/twc_tcw.nc"
PATH_tp="data/tp_tp.nc"
PATH_temp="data/temp_t2m.nc"
PATH_slp="data/slp_msl.nc"
PATH_cpe="data/cpe_cape.nc"
PATH_obs="obsdata/"

TCW=Dataset(PATH_tcw,more="r")
print(TCW.variables["tcw"].shape)
TP=Dataset(PATH_tp,more="r")
TEMP=Dataset(PATH_temp,more="r")
SLP=Dataset(PATH_slp,more="r")
CPE=Dataset(PATH_cpe,more="r")

files = glob.glob(os.path.join(PATH_obs, "*.nc"))
Highres= [Dataset(file,more="r") for file in files]

mask=ma.getmask(Highres[0].variables["RAINFALL"][0][2:,3:-4])
Dataobs=[[],[],[],[],[]]
for j in range(len(Highres)):#len(Highres)
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
	
		Dataobs[0].append(D[0:48,20:68])
		Dataobs[1].append(D[40:88,0:48])
		Dataobs[2].append(D[40:88,40:88])
		Dataobs[3].append(D[50:98,80:128])
		Dataobs[4].append(D[80:128,15:63])

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

def regiontransform(region):
	region=tf.convert_to_tensor(region)
	region=tf.expand_dims(region, axis=-1)
	Pooling_layer=tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
	region=Pooling_layer(region)
	region=Pooling_layer(region)
	region=tf.expand_dims(region,axis=1)
	region=tf.reshape(region,(-1,24,12,12,1))
	region=tf.cast(region,tf.float32)
	return region   

def data_generator(NETCDF_file, mask, variable):
	Data=[[],[],[],[],[]]
	for i in range(NETCDF_file.variables[variable].shape[0]):
		
		Dp=np.array(NETCDF_file.variables[variable][i,:,:])
		Dp=np.flip(Dp, axis=0)
		if (np.amax(Dp)!=0):
            		Dp=Dp/np.amax(Dp)
		Dp=Dp[:-1,1:]

		for j in range(mask.shape[0]):
			for k in range(mask.shape[1]):
				if mask[j][k]== True:
					Dp[j][k]=-1
		Dp[Dp.shape[0]-1][:]=-1
		#Dp=ma.masked_less(Dp,0)
	
		Data[0].append(Dp[0:48, 20:68])
		Data[1].append(Dp[40:88,0:48])
		Data[2].append(Dp[40:88,40:88])
		Data[3].append(Dp[30:78,80:128])
		Data[4].append(Dp[80:128,15:63])
	
	north=regiontransform(Data[4])	
	west=regiontransform(Data[1])
	central=regiontransform(Data[2])
	east=regiontransform(Data[3])
	south=regiontransform(Data[0])
	return [north, west, central, east, south]
	
TCWDATA=data_generator(TCW, mask, variable="tcw")
print("TCWDATA done")
TPDATA=data_generator(TP, mask, variable="tp")
print("TPDATA done")
#SLPDATA=data_generator(SLP, mask, variable="msl")
#print("SLPDATA done")
#CPEDATA=data_generator(CPE, mask, variable="cape")
#print("CPEDATA done")
TEMPDATA=data_generator(TEMP, mask, variable="t2m")
print("TEMPDATA done")

Final_north=Concatenate(axis=-1)([TCWDATA[0],TPDATA[0],TEMPDATA[0]])
Final_west=Concatenate(axis=-1)([TCWDATA[1],TPDATA[1],TEMPDATA[1]])
Final_central=Concatenate(axis=-1)([TCWDATA[2],TPDATA[2],TEMPDATA[2]])
Final_east=Concatenate(axis=-1)([TCWDATA[3],TPDATA[3],TEMPDATA[3]])
Final_south=Concatenate(axis=-1)([TCWDATA[4],TPDATA[4],TEMPDATA[4]])
print(Final_north.shape,Final_central.shape,Final_east.shape,Final_south.shape)
Input_data=Concatenate(axis=0)([Final_north,Final_west,Final_central,Final_east,Final_south])

#Input_data=Concatenate(axis=0)([TPDATA[0],TPDATA[1],TPDATA[2],TPDATA[3],TPDATA[4]])
labelfile=np.array([i for i in range(5) for j in range(northobs.shape[0])])#Highres_data.shape[0]
labelfile=np.expand_dims(labelfile, axis=-1)
print(Input_data.shape)
np.save("Inputfile3channels", Input_data)
np.save("Highresfile3channels", Highres_data)
np.save("labelfile3channels",labelfile)


#long1=NETCDF_DATA.variables["longitude"][1:]
#lat1=NETCDF_DATA.variables["latitude"][1:]
#(ln1, lt1) = (long1[15:63], lat1[0:48])
#(ln2, lt2) = (long1[0:48], lat1[40:88])
#(ln3, lt3) = (long1[40:88], lat1[40:88])
#(ln4, lt4) = (long1[80:128], lat1[30:78])
#(ln5, lt5)=(long1[20:68], lat1[80:128])








#plt.style.use("seaborn-v0_8-bright")
#figure = plt.figure(figsize=(10,10))
#axis_func = plt.axes(projection=ccrs.PlateCarree())
#axis_func.coastlines(resolution="10m",linewidth=1)
#axis_func.gridlines(linestyle='--',color='black',linewidth=2)
#N=99
#plt.contourf(ln4, lt4, Dp4, N, transform=ccrs.PlateCarree(), cmap='terrain')
#color_bar_func = plt.colorbar(ax=axis_func, orientation="vertical", pad=0.05, aspect=16, shrink=.8)
#color_bar_func.ax.tick_params(labelsize=8)
#plt.tight_layout()
#plt.savefig('dp4.png')
#plt.show()

#plt.style.use("seaborn-v0_8-bright")
#figure = plt.figure(figsize=(10,10))
#axis_func = plt.axes(projection=ccrs.PlateCarree())
#axis_func.coastlines(resolution="10m",linewidth=1)
#axis_func.gridlines(linestyle='--',color='black',linewidth=2)
#N=99
#plt.contourf(long2, lat2, plotd, N, transform=ccrs.PlateCarree(), cmap='terrain')
#color_bar_func = plt.colorbar(ax=axis_func, orientation="vertical", pad=0.05, aspect=16, shrink=.8)
#color_bar_func.ax.tick_params(labelsize=8)
#plt.tight_layout()
#plt.savefig('imd.png')
#plt.show()




#order of operation
#segment into 5 sections over india
# collect the similar segments in a individual array
# group the 24 elements using expnand dims
#reduce the resolution of each
# repeat the process for other 4 fields (create a function)
# use concat to make 5 channel input of respective segments of India
#merge all the five segments to create the final dataset
