# Developed by Bhargav Purohit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import os as os


scalar=StandardScaler()

Weight_KG=np.array([60,75,55,66,68])
Weight_GM=Weight_KG*1000
Height=np.array([152,160,162,170,149])

DF_KG=pd.DataFrame({"Weight_KG": Weight_KG, 
                      "Height": Height})

DF_GM=pd.DataFrame({"Weight_GM": Weight_GM, 
                      "Height": Height})
                        

for dataset in [DF_KG, DF_GM]:
    print("CHECKING ORIGINAL DATA SET")
    distances = pdist(dataset.values, metric='euclidean')
    dist_matrix = squareform(distances)
    DF=pd.DataFrame(dist_matrix)
    for ind in DF.index:
        arr=DF.values[ind]
        min=arr[arr!=0].min()
        pos=int(np.where(arr==min)[0])
        print("{} is close to {} with euclidien value {}".format(ind, pos, min))
        
        
 
 from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()

DF_KG["Weight_KG_Scaled"] = scalar.fit_transform(DF_KG[["Weight_KG"]])
DF_KG["Height_Scaled"] = scalar.fit_transform(DF_KG[["Height"]])
distances = pdist(DF_KG[["Weight_KG_Scaled", "Height_Scaled"]].values, metric='euclidean')
dist_matrix = squareform(distances)
print("Scaled Data set(KG)")
print(pd.DataFrame(dist_matrix))



DF_GM["Weight_GM_Scaled"] = scalar.fit_transform(DF_GM[["Weight_GM"]])
DF_GM["Height_Scaled"] = scalar.fit_transform(DF_GM[["Height"]])
distances = pdist(DF_GM[["Weight_GM_Scaled", "Height_Scaled"]].values, metric='euclidean')
dist_matrix = squareform(distances)
print("Scaled Data set(GM)")
print(pd.DataFrame(dist_matrix))
