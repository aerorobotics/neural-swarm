import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file", help="csv file")
args = parser.parse_args()

data = np.loadtxt(args.file, delimiter=',', skiprows=1)
print(data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,1], data[:,2], data[:,0])
plt.show()