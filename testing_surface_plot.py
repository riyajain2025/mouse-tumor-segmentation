# imports 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# functions 

def read_data(file_name):
    with open(file_name, 'rb') as fid:
        data = np.fromfile(fid, dtype='>f8')
    
    points = data.reshape(-1, 3)
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    return x, y, z

def interpolate_z(x, y, z):
    grid_x, grid_y = np.meshgrid(
          np.linspace(min(x), max(x), 100),
          np.linspace(min(y), max(y), 100)
     )
    
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

    return grid_x, grid_y, grid_z

file_path = ''
x, y, z = read_data('./data/invotive_data/L3.3 Parental 2-Control-1-35-38.bin')
x, y, z = interpolate_z(x, y, z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, cmap='viridis')

plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.show()