import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
%matplotlib inline
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


Normal=pd.DataFrame({"A": np.random.normal(0,5,100),
                 "B": np.random.normal(5,7,100)
                }
               )
Outliers=pd.DataFrame({"A": np.random.randint(30,40,5),
                       "B": np.random.randint(40,60,5)
                    
                      }
                     )

NEW_DF=pd.concat([Normal, Outliers])
#NEW_DF=Normal
scalar=StandardScaler()
NEW_DF["Scaled_A"] = scalar.fit_transform(NEW_DF[["A"]])
NEW_DF["Scaled_B"] = scalar.fit_transform(NEW_DF[["B"]])

robust=RobustScaler()
NEW_DF["Robust_A"] = robust.fit_transform(NEW_DF[["A"]])
NEW_DF["Robust_B"] = robust.fit_transform(NEW_DF[["B"]])

MinMax=MinMaxScaler()
NEW_DF["MinMax_A"] = MinMax.fit_transform(NEW_DF[["A"]])
NEW_DF["MinMax_B"] = MinMax.fit_transform(NEW_DF[["B"]])


fig, (axis1,axis2, axis3, axis4) = plt.subplots(1,4, figsize=(15, 5))
axis1.title.set_text('Before scaling')
NEW_DF[["A", "B"]].boxplot(ax=axis1);
axis2.title.set_text('After standardization')
NEW_DF[["Scaled_A", "Scaled_B"]].boxplot(ax=axis2);
axis3.title.set_text('After RobustScaling')
NEW_DF[["Robust_A", "Robust_B"]].boxplot(ax=axis3);
axis4.title.set_text('After MinMaxScaling')
NEW_DF[["MinMax_A", "MinMax_B"]].boxplot(ax=axis4);
