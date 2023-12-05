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
from netCDF4 import Dataset
import cartopy.crs as ccrs
from tensorflow import keras
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Add
from keras.models import load_model
import numpy.ma as ma
import os
import glob
import pysteps

PATH_model="pretrain_2_3field"
PATH_files="validation_data\high_res"
models=glob.glob(os.path.join(PATH_model, "*.keras"))
files = glob.glob(os.path.join(PATH_files, "*.nc"))
INPUT=np.load("3Fieldtest.npy")
OUTPUT=np.load("3Fieldhighrestest.npy")
modelslist=[load_model(model, compile=False) for model in models]
#model=load_model("WGAN_GP_GENLOSS.keras")
Scale_data=[Dataset(file, more="r") for file in files]
#print(modelslist)
#IMD_DATA=Dataset("RF25_ind2021_rfp25.nc", more="r")
#ERA_DATA=Dataset("tptest.nc", more="r")
scale_factor=[]
print(INPUT.shape, OUTPUT.shape)
for i in range(len(Scale_data)):
	if (Scale_data[i].variables["RAINFALL"].shape[0]==365):
		loop=range(152,274,1)
	else:
		loop=range(153,275,1)
	for j in loop:
		Dp=np.array(Scale_data[i].variables["RAINFALL"][j,:,:])
		Dp[Dp<=0]=0
		scale_factor.append(np.amax(Dp))
print(len(scale_factor))
scale_input=5*scale_factor
print(len(scale_input))

def FSS_func(modelslist, amount, window): 
  Results=[]
  for model in modelslist:
  	FSS=[]
  	for i in range(INPUT.shape[0]):
      		imd=OUTPUT[i]
      		imd=np.reshape(imd,(48,48))
      		imd[imd<=0]=0
      		imd=imd*scale_input[i]
      		if (np.amax(imd)>amount):
        		gen=INPUT[i]
        		gen=np.reshape(gen, (1,24,12,12,3))
        		OUT=model.predict(gen, verbose=0)
        		OUT=np.reshape(OUT,(48,48))
        		OUT[OUT<=0]=0
        		OUT=OUT*scale_input[i]
        		FSS.append(pysteps.verification.spatialscores.fss(OUT, imd,amount, window))
  	FSSMean=(np.mean(np.array(FSS)))
  	FSSSD=(np.std(np.array(FSS)))
  	Results.append([FSSMean, FSSSD])
  	#print(FSSMean, FSSSD)
  return Results

#Results=FSS_func(modelslist, amount=1, window=2) 
#print("results for amount=1 and window =2")
#print(Results)

#Results=FSS_func(modelslist, amount=1, window=1) #modelslist
#print("results for amount=1 and window =1")
#print(Results)

#Results=FSS_func(modelslist, amount=10, window=2) #modelslist
#print("results for amount=10 and window =2")
#print(Results)

#Results=FSS_func(modelslist, amount=10, window=1) #modelslist
#print("results for amount=10 and window =1")
#print(Results)

Results=FSS_func(modelslist, amount=50, window=2) #modelslist
print("results for amount=50 and window =2")
print(Results)

Results=FSS_func(modelslist, amount=50, window=1) #modelslist
print("results for amount=50 and window =1")
print(Results)

Results=FSS_func(modelslist, amount=64.4, window=1) #modelslist
print("results for amount=64.4 and window =1")
print(Results)