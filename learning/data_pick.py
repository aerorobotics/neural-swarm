import matplotlib.pyplot as plt
from utils import data_extraction

# This file is for data checking and pick
# Right now need to manully export something like "datacollection19_summary"
# Will improve later
def vis(Data1, Data2, ss1, ee1, ss2, ee2):
    plt.figure(figsize=(12,4))
    plt.subplot(1, 3, 1)
    line1, = plt.plot(Data1['time'][ss1:ee1], Data1['pos'][ss1:ee1, 0])
    line2, = plt.plot(Data1['time'][ss1:ee1], Data1['pos'][ss1:ee1, 1])
    line3, = plt.plot(Data1['time'][ss1:ee1], Data1['pos'][ss1:ee1, 2])
    plt.legend([line1, line2, line3], ['x', 'y', 'z'])
    plt.grid()
    plt.title('traj (agent a)')
    plt.subplot(1, 3, 2)
    line1, = plt.plot(Data2['time'][ss2:ee2], Data2['pos'][ss2:ee2, 0])
    line2, = plt.plot(Data2['time'][ss2:ee2], Data2['pos'][ss2:ee2, 1])
    line3, = plt.plot(Data2['time'][ss2:ee2], Data2['pos'][ss2:ee2, 2])
    plt.legend([line1, line2, line3], ['x', 'y', 'z'])
    plt.grid()
    plt.title('traj (agent b)')
    plt.subplot(1, 3, 3)
    plt.scatter(Data1['pos'][ss1:ee1, 1], Data1['pos'][ss1:ee1, 2])
    plt.scatter(Data2['pos'][ss2:ee2, 1], Data2['pos'][ss2:ee2, 2])
    plt.legend(['a', 'b'])
    plt.show()
    print("a: t0, t1 = " + str(Data1['time'][ss1]) + ', ' + str(Data1['time'][ee1]))
    print("b: t0, t1 = " + str(Data2['time'][ss2]) + ', ' + str(Data2['time'][ee2]))

Data_a = data_extraction('../../datacollection19_12_11_2019_2/swap_large_small/cf102_01.csv')
Data_b = data_extraction('../../datacollection19_12_11_2019_2/swap_large_small/cf50_01.csv')
vis(Data_a, Data_b, 0, -1, 0, -1)