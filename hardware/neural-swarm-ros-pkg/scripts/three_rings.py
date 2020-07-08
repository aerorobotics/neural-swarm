#!/usr/bin/env python

import numpy as np

from pycrazyswarm import *
import pycrazyswarm.cfsim.cffirmware as firm
import uav_trajectory
import colorsys

TIMESCALE = 1.0

# CF -> trajId
# traj 1 - 6: horizontal ring
# traj 7 - 11: vertical (negative x)
# traj 12 - 16: vertical (positive x)
root = 'three_rings16_pps'
TrajIds = {
   1: 12,
   4: 13,
   5: 1,
   # 50: 15,
   200: 15,
   6: 16,

   7: 2,
   8: 14,
   101: 6,
   51: 4,
   102: 3,
   9: 7,

   10: 5,
   52: 8,
   11: 9,
   17: 10,
   21: 11,
}

OFFSET = [0.0, 0.0, 0.0]

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # upload trajectories and initialize color
    hues = np.linspace(0,1.0,len(allcfs.crazyflies),endpoint=False)
    T = 0
    for cf, hue in zip(allcfs.crazyflies, hues):
        fname = '{0}/pp{1}.csv'.format(root, TrajIds[cf.id])
        traj = uav_trajectory.Trajectory()
        traj.loadcsv(fname)
        cf.uploadTrajectory(0, 0, traj)
        cf.startPos = traj.eval(0).pos
        T = max(T, traj.duration)

        r,g,b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        cf.setParam("ring/effect", 0)#7) # solid color
        cf.setParam("ring/solidRed", int(r * 255))
        cf.setParam("ring/solidGreen", int(g * 255))
        cf.setParam("ring/solidBlue", int(b * 255))


    print("T: ", T * TIMESCALE)

    # enable ground effect in the NN computation
    allcfs.setParam("ctrlFa/enableGround", 1)
    allcfs.setParam("ctrlFa/filter", 0.2)
    # enable NN
    allcfs.setParam("ctrlFa/enableNN", 0)

    allcfs.takeoff(targetHeight=0.7, duration=2.5)
    timeHelper.sleep(3.0)

    for cf in allcfs.crazyflies:
        pos = cf.startPos
        cf.goTo(np.array(pos) + np.array(OFFSET), 0, 3.0)
    timeHelper.sleep(3.5)

    allcfs.setParam("usd/logging", 1)

    allcfs.startTrajectory(0, timescale=TIMESCALE)
    timeHelper.sleep(T * TIMESCALE + 1.0)
    # allcfs.startTrajectory(0, timescale=TIMESCALE, reverse=True)
    # timeHelper.sleep(T * TIMESCALE + 1.0)

    allcfs.setParam("usd/logging", 0)
    allcfs.setParam("ring/effect", 0) # disable LED ring

    for cf in allcfs.crazyflies:
        pos = np.array(cf.initialPosition) + np.array([0, 0, 0.7])
        cf.goTo(pos, 0, 3.0)
    timeHelper.sleep(3.5)

    allcfs.land(targetHeight=0.06, duration=2.5)
    timeHelper.sleep(3.0)

