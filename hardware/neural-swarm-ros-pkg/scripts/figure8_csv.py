#!/usr/bin/env python

import numpy as np

from pycrazyswarm import *
import uav_trajectory

TIMESCALES = [1.0, 0.8]

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    traj1 = uav_trajectory.Trajectory()
    traj1.loadcsv("figure8.csv")

    for cf in allcfs.crazyflies:
        cf.uploadTrajectory(0, 0, traj1)

    allcfs.takeoff(targetHeight=1.0, duration=2.0)
    timeHelper.sleep(2.5)
    for cf in allcfs.crazyflies:
        pos = np.array(cf.initialPosition) + np.array([0, 0, 1.0])
        cf.goTo(pos, 0, 2.0)
    timeHelper.sleep(2.5)

    allcfs.setParam("usd/logging", 1)

    for timescale in TIMESCALES:

        allcfs.startTrajectory(0, timescale=timescale)
        timeHelper.sleep(traj1.duration * timescale + 0.5)
        allcfs.startTrajectory(0, timescale=timescale, reverse=True)
        timeHelper.sleep(traj1.duration * timescale + 0.5)

    allcfs.setParam("usd/logging", 0)

    allcfs.land(targetHeight=0.06, duration=2.0)
    timeHelper.sleep(3.0)

