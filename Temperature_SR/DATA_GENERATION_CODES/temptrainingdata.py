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



PATH_IMD = "IMD_DATA_MAX"
PATH_ERA="ERA_DATA"
PATH_mask="RF25_ind2021_rfp25.nc"

files = glob.glob(os.path.join(PATH_ERA, "*.nc"))
ERADATA= [Dataset(file,more="r") for file in files]

files = glob.glob(os.path.join(PATH_IMD, "*.nc"))
IMDDATA= [Dataset(file,more="r") for file in files]

maskdata=Dataset(PATH_mask, more="r")
mask=ma.getmask(maskdata.variables["RAINFALL"][0][2:,3:-4])

def regiontransform(region):
	region=tf.convert_to_tensor(region)
	region=tf.expand_dims(region, axis=-1)
	region=tf.cast(region,tf.float32)
	return region

Dataobs=[[],[],[],[],[]]
Hourlymax=np.zeros(shape=(128,128))
#for j in range(len(ERADATA)):
for i in range(ERADATA[2].variables["t2m"].shape[0]-(1095*24),ERADATA[2].variables["t2m"].shape[0],1):
	Dt=np.array(ERADATA[2].variables["t2m"][i,:,:])
	Dt=np.flip(Dt, axis=0)
	Dt=Dt-273.15
	for k in range(Dt.shape[0]):
		for l in range(Dt.shape[1]):
			if (Dt[k][l]>=Hourlymax[k][l]):
				Hourlymax[k][l]=Dt[k][l]
		
	if (i%24==23):
		Dp=Hourlymax
		Hourlymax=np.zeros(shape=(128,128))			
		Dp=Dp[:-1][:]
		for k in range(mask.shape[0]):
			for l in range(mask.shape[1]):
				if mask[k][l]== True:
					Dp[k][l]=51
		npad=((0,1),(0,0))
		Dp=np.pad(Dp, pad_width=npad, mode='constant', constant_values=51)
		Dp=Dp/51.0	
		Dataobs[0].append(Dp[0:48,20:68]) #south
		Dataobs[1].append(Dp[40:88,0:48]) #west
		Dataobs[2].append(Dp[40:88,40:88]) #central
		Dataobs[3].append(Dp[48:96,80:128]) #east
		Dataobs[4].append(Dp[80:128,16:64]) #north
		
northobs= regiontransform(Dataobs[4])
westobs= regiontransform(Dataobs[1])
centralobs= regiontransform(Dataobs[2])
eastobs= regiontransform(Dataobs[3])
southobs= regiontransform(Dataobs[0])
ERA_data=Concatenate(axis=0)([northobs,westobs,centralobs,eastobs,southobs])
print(ERA_data.shape)
np.save("Highresfiletemp2017-2019", ERA_data)

Datain=[[],[],[],[],[]]
for j in range(len(IMDDATA)-3,len(IMDDATA),1):
	for i in range(IMDDATA[j].variables["max_temp"].shape[0]):
		imddp=np.array(IMDDATA[j].variables["max_temp"][i,:,:])
		npad=((0,1),(0,1))
		imddp=np.pad(imddp, pad_width=npad, mode='constant', constant_values=51.0)
		imddp[imddp>50]=51.0
		imddp=imddp/51.0
		#imddp=np.ma.masked_greater(imddp,51)
		Datain[0].append(imddp[0:12,5:17])   #south
		Datain[1].append(imddp[10:22,0:12])  #west
		Datain[2].append(imddp[10:22,10:22]) #central
		Datain[3].append(imddp[12:24,20:32]) #east 
		Datain[4].append(imddp[20:32,4:16]) #north

   

north= regiontransform(Datain[4])
west= regiontransform(Datain[1])
central= regiontransform(Datain[2])
east= regiontransform(Datain[3])
south= regiontransform(Datain[0])
Input_data=Concatenate(axis=0)([north,west,central,east,south])
print(Input_data.shape)
np.save("Inputfiletemp2017-2019", Input_data)




