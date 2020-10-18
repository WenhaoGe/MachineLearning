import pandas as pd
import numpy as np
from numpy import zeros,ones,eye
import matplotlib.pyplot as plt
import random
import math
from sklearn.preprocessing import Imputer

filename = "proj1Dataset.xlsx"
df = pd.read_excel(filename)
Weight = np.array(df["Weight"])
Weight = Weight/np.max(Weight)
HP = df["Horsepower"]

onesArr = np.ones(406)
Weight = np.column_stack((Weight, onesArr))
HP_average = HP.mean()
HP.fillna(HP_average,inplace=True)
HP = np.array(HP)
x = Weight.T @ Weight
ainv = np.linalg.pinv(x)
result = ainv @ Weight.T @ HP

plt.scatter(Weight[:,0],HP)
plt.plot(Weight[:,0],Weight@result)

#gradient descent method
x1 = random.random()
my_arr = []
my_arr.append(x1)
x2 = random.random()
my_arr.append(x2)
rho = 1e-3
my_arr = np.array(my_arr)
for i in range(5000):
    J = 2 * my_arr.T @ Weight.T @ Weight - 2 * HP.T @ Weight
    my_arr = my_arr - rho * J
    
plt.figure()
plt.scatter(Weight[:,0],HP)
plt.plot(Weight[:,0],Weight@my_arr)