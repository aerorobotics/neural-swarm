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

# predict using the model
X = data[:, 1] / 65536 # PWM
Y = data[:, 2] / 4.2 # Vol
# Z = data[:, 0] / 4 # Thrust (gram)

C_00 = 11.1
C_10 = -39.1
C_01 = -9.5
C_20 = 20.6
C_11 = 38.4

force = C_00 + C_10 * X + C_01 * Y + C_20 * X * X + C_11 * X * Y
ax.scatter(data[:,1], data[:,2], force * 4)

plt.show()