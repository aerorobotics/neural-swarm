#!/usr/bin/env python

import numpy as np
import random
import yaml

from pycrazyswarm import *
import uav_trajectory

configs = [
    # {
    #     "name": "Ge2S",
    #     "min_height": 0.05,
    #     "ids": [51]
    # },
    # {
    #     "name": "Ge2L",
    #     "min_height": 0.06,
    #     "ids": [101]
    # },

    # {
    #     "name": "S2S",
    #     "min_height": 0.5,
    #     "ids": [50, 51]
    # },
    # {
    #     "name": "S2L",
    #     "min_height": 0.5,
    #     "ids": [101, 50]
    # },
    # {
    #     "name": "L2S",
    #     "min_height": 0.5,
    #     "ids": [50, 101]
    # },
    # {
    #     "name": "L2L",
    #     "min_height": 0.5,
    #     "ids": [102, 101]
    # },

    # {
    #     "name": "SS2S",
    #     "min_height": 0.5,
    #     "ids": [52, 51, 50]
    # },
    # {
    #     "name": "SL2S",
    #     "min_height": 0.5,
    #     "ids": [52, 51, 101]
    # },
    # {
    #     "name": "LL2S",
    #     "min_height": 0.5,
    #     "ids": [50, 101, 102]
    # },
    # {
    #     "name": "SS2L",
    #     "min_height": 0.5,
    #     "ids": [101, 51, 50]
    # },
    # {
    #     "name": "SL2L",
    #     "min_height": 0.5,
    #     "ids": [101, 51, 102]
    # },
    # {
    #     "name": "LL2L",
    #     "min_height": 0.7,
    #     "ids": [102, 101, 100]
    # },

    # {
    #     "name": "SSS2S",
    #     "min_height": 0.5,
    #     "ids": [50, 51, 52, 2]
    # },

    {
        "name": "S2S (motor upgrade)",
        "min_height": 0.5,
        "ids": [200, 50]
    },


]

SwapTimes = [4,3]
Radius = 0.25

def is_small(cfid):
    if cfid < 100 or cfid >= 200:
        return True
    return False

if __name__ == "__main__":

    with open('validation_config.yaml', 'w') as f:
        yaml.dump(configs, f)

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # enable ground effect in the NN computation
    allcfs.setParam("ctrlFa/enableGround", 1)
    allcfs.setParam("ctrlFa/filter", 0.35)

    for config in configs:
        print("Executing: ", config["name"])
        Ids = config["ids"]
        Min_Height = config["min_height"]

        if len(Ids) == 1:
            # disable NN by default
            allcfs.setParam("ctrlFa/enableNN", 0)
            allcfs.setParam("ctrlSJC/trigReset", 1) # reset all of the state

            # only take-off selected CFs (no broadcast!)
            for cfid in Ids:
                allcfs.crazyfliesById[cfid].takeoff(targetHeight=0.5, duration=3.0)

            timeHelper.sleep(3)

            allcfs.setParam("usd/logging", 1)
            for useNN in [0, 2]:
                allcfs.setParam("ctrlFa/enableNN", useNN)
                allcfs.setParam("ctrlSJC/trigReset", 2) # set z-error to 0
                timeHelper.sleep(5.0) # let the integral controller settle...

                for duration in [3.5,2.5,1.5]:
                    # move close to ground
                    for cfid in Ids:
                        cf = allcfs.crazyfliesById[cfid]
                        pos = np.array(cf.initialPosition) + np.array([0, 0, Min_Height])
                        cf.goTo(pos, 0, duration)
                    timeHelper.sleep(duration + 0.5)

                    # move up again ground
                    for cfid in Ids:
                        cf = allcfs.crazyfliesById[cfid]
                        pos = np.array(cf.initialPosition) + np.array([0, 0, 0.5])
                        cf.goTo(pos, 0, duration)
                    timeHelper.sleep(duration + 0.5)
            allcfs.setParam("usd/logging", 0)

            # land flying CFs
            for cfid in Ids:
                allcfs.crazyfliesById[cfid].land(targetHeight=0.02, duration=3.0)
            timeHelper.sleep(5.0)

        else:

            Heights = [Min_Height]
            for i in range(len(Ids)-1):
                if is_small(Ids[i]) and is_small(Ids[i+1]):
                    # both are small CFs
                    Heights.append(Heights[i] + 0.3)
                elif is_small(Ids[i]) and not is_small(Ids[i+1]):
                    # below is small, top is large
                    Heights.append(Heights[i] + 0.5)
                elif not is_small(Ids[i]) and is_small(Ids[i+1]):
                    # below is large, top is small
                    Heights.append(Heights[i] + 0.2)
                elif not is_small(Ids[i]) and not is_small(Ids[i+1]):
                    # both are large
                    Heights.append(Heights[i] + 0.3)

            # computing starting point angles
            angles = np.linspace(np.pi/2, 2.5*np.pi, len(Ids), endpoint=False)

            # disable NN by default
            allcfs.setParam("ctrlFa/enableNN", 0)
            allcfs.setParam("ctrlSJC/trigReset", 1) # reset all of the state

            # only take-off selected CFs (no boradcast!)
            for cfid in Ids:
                allcfs.crazyfliesById[cfid].takeoff(targetHeight=np.min(Heights), duration=3.0)

            timeHelper.sleep(3.5)

            # go to initial positions
            for angle, cfid, height in zip(angles[0:len(Ids)], Ids, Heights):
                pos = np.array([np.cos(angle) * Radius, np.sin(angle) * Radius, height])
                allcfs.crazyfliesById[cfid].goTo(pos, 0, 3.0)

            timeHelper.sleep(3.5)

            allcfs.setParam("usd/logging", 1)

            for useNN in [0, 2]:
            # for useNN in [2]:
                allcfs.setParam("ctrlFa/enableNN", useNN)
                allcfs.setParam("ctrlSJC/trigReset", 2) # set z-error to 0
                timeHelper.sleep(5.0) # let the integral controller settle...

                for swapTime in SwapTimes:
                    # swap 1
                    for angle, cfid, height in zip(angles, Ids, Heights):
                        pos = np.array([np.cos(angle + np.pi) * Radius, np.sin(angle + np.pi) * Radius, height])
                        allcfs.crazyfliesById[cfid].goTo(pos, 0, swapTime)
                    timeHelper.sleep(swapTime+0.1)

                    # swap 2
                    for angle, cfid, height in zip(angles, Ids, Heights):
                        pos = np.array([np.cos(angle) * Radius, np.sin(angle) * Radius, height])
                        allcfs.crazyfliesById[cfid].goTo(pos, 0, swapTime)
                    timeHelper.sleep(swapTime+0.1)

            allcfs.setParam("usd/logging", 0)

            for cfid in Ids:
                cf = allcfs.crazyfliesById[cfid]
                pos = np.array(cf.initialPosition) + np.array([0, 0, np.min(Heights)])
                cf.goTo(pos, 0, 3.0)
            timeHelper.sleep(3.5)

            # land flying CFs
            for cfid in Ids:
                allcfs.crazyfliesById[cfid].land(targetHeight=0.02, duration=3.0)
            timeHelper.sleep(5.0)
