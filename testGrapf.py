import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pandas as pd
import matplotlib.cm as cm


df = pd.read_csv ('irisdata.txt', skiprows=9, header = None, delim_whitespace=True)

print(df)

sequence_containing_x_vals = df.iloc [:,0]
sequence_containing_y_vals = df.iloc [:,1]
sequence_containing_z_vals = df.iloc [:,2]

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter3D(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
plt.show()