#!/usr/bin/env python

import numpy as np
import random
import yaml

from pycrazyswarm import *
import uav_trajectory


configs = [
    # {
    #     "name": "S2S",
    #     "heights": [0.5, 0.8],
    #     "ids": [50, 51],
    #     "firmware_params":
    #     {
    #         "motorPowerSet/mixMode": 1,
    #         # attitude lambda
    #         "ctrlSJC/Katt_Px": 10,
    #         "ctrlSJC/Katt_Py": 10,
    #         "ctrlSJC/Katt_Pz": 4,
    #         # attitude K
    #         "ctrlSJC/Katt_Dx": 0.002,
    #         "ctrlSJC/Katt_Dy": 0.002,
    #         "ctrlSJC/Katt_Dz": 0.001,
    #         # delay
    #         "ctrlSJC/Katt_Dwx": 0.00008,
    #         "ctrlSJC/Katt_Dwy": 0.00008,
    #         "ctrlSJC/Katt_Dwz": 0.00002,
    #         # attitude I
    #         "ctrlSJC/Katt_Ix": 0.002,
    #         "ctrlSJC/Katt_Iy": 0.002, 
    #         "ctrlSJC/Katt_Iz": 0.004,
    #         "ctrlSJC/Katt_I_limit": 2,
    #     }
    # },

    # {
    #     "name": "S2S",
    #     "heights": [0.5, 0.8],
    #     "ids": [50, 51],
    #     "firmware_params":
    #     {
    #         "motorPowerSet/mixMode": 1,
    #         # attitude lambda
    #         "ctrlSJC/Katt_Px": 10,
    #         "ctrlSJC/Katt_Py": 10,
    #         "ctrlSJC/Katt_Pz": 4,
    #         # attitude K
    #         "ctrlSJC/Katt_Dx": 0.002,
    #         "ctrlSJC/Katt_Dy": 0.002,
    #         "ctrlSJC/Katt_Dz": 0.001,
    #         # delay
    #         "ctrlSJC/Katt_Dwx": 0.00008,
    #         "ctrlSJC/Katt_Dwy": 0.00008,
    #         "ctrlSJC/Katt_Dwz": 0.00002,
    #         # attitude I
    #         "ctrlSJC/Katt_Ix": 0.002,
    #         "ctrlSJC/Katt_Iy": 0.002, 
    #         "ctrlSJC/Katt_Iz": 0.004,
    #         "ctrlSJC/Katt_I_limit": 0,
    #     }
    # },

    # {
    #     "name": "S2S",
    #     "heights": [0.5, 0.8],
    #     "ids": [50, 51],
    #     "firmware_params":
    #     {
    #         "motorPowerSet/mixMode": 1,
    #         # attitude lambda
    #         "ctrlSJC/Katt_Px": 5,
    #         "ctrlSJC/Katt_Py": 5,
    #         "ctrlSJC/Katt_Pz": 2,
    #         # attitude K
    #         "ctrlSJC/Katt_Dx": 0.002,
    #         "ctrlSJC/Katt_Dy": 0.002,
    #         "ctrlSJC/Katt_Dz": 0.001,
    #         # delay
    #         "ctrlSJC/Katt_Dwx": 0.00008,
    #         "ctrlSJC/Katt_Dwy": 0.00008,
    #         "ctrlSJC/Katt_Dwz": 0.00002,
    #         # attitude I
    #         "ctrlSJC/Katt_Ix": 0.002,
    #         "ctrlSJC/Katt_Iy": 0.002, 
    #         "ctrlSJC/Katt_Iz": 0.004,
    #         "ctrlSJC/Katt_I_limit": 2,
    #     }
    # },

    # {
    #     "name": "S2S",
    #     "heights": [0.5, 0.8],
    #     "ids": [50, 51],
    #     "firmware_params":
    #     {
    #         "motorPowerSet/mixMode": 1,
    #         # attitude lambda
    #         "ctrlSJC/Katt_Px": 5,
    #         "ctrlSJC/Katt_Py": 5,
    #         "ctrlSJC/Katt_Pz": 2,
    #         # attitude K
    #         "ctrlSJC/Katt_Dx": 0.001,
    #         "ctrlSJC/Katt_Dy": 0.001,
    #         "ctrlSJC/Katt_Dz": 0.0005,
    #         # delay
    #         "ctrlSJC/Katt_Dwx": 0.00008,
    #         "ctrlSJC/Katt_Dwy": 0.00008,
    #         "ctrlSJC/Katt_Dwz": 0.00002,
    #         # attitude I
    #         "ctrlSJC/Katt_Ix": 0.002,
    #         "ctrlSJC/Katt_Iy": 0.002, 
    #         "ctrlSJC/Katt_Iz": 0.004,
    #         "ctrlSJC/Katt_I_limit": 2,
    #     }
    # },

    # {
    #     "name": "S2S",
    #     "heights": [0.5, 0.8],
    #     "ids": [50, 51],
    #     "firmware_params":
    #     {
    #         "motorPowerSet/mixMode": 1,
    #         # attitude lambda
    #         "ctrlSJC/Katt_Px": 5,
    #         "ctrlSJC/Katt_Py": 5,
    #         "ctrlSJC/Katt_Pz": 2,
    #         # attitude K
    #         "ctrlSJC/Katt_Dx": 0.001,
    #         "ctrlSJC/Katt_Dy": 0.001,
    #         "ctrlSJC/Katt_Dz": 0.0005,
    #         # delay
    #         "ctrlSJC/Katt_Dwx": 0.00004,
    #         "ctrlSJC/Katt_Dwy": 0.00004,
    #         "ctrlSJC/Katt_Dwz": 0.00001,
    #         # attitude I
    #         "ctrlSJC/Katt_Ix": 0.002,
    #         "ctrlSJC/Katt_Iy": 0.002, 
    #         "ctrlSJC/Katt_Iz": 0.004,
    #         "ctrlSJC/Katt_I_limit": 0,
    #     }
    # },

    # {
    #     "name": "S2S",
    #     "heights": [0.5, 0.8],
    #     "ids": [50, 51],
    #     "firmware_params":
    #     {
    #         "motorPowerSet/mixMode": 1,
    #         # attitude lambda
    #         "ctrlSJC/Katt_Px": 8,
    #         "ctrlSJC/Katt_Py": 8,
    #         "ctrlSJC/Katt_Pz": 4,
    #         # attitude K
    #         "ctrlSJC/Katt_Dx": 0.001,
    #         "ctrlSJC/Katt_Dy": 0.001,
    #         "ctrlSJC/Katt_Dz": 0.001,
    #         # delay
    #         "ctrlSJC/Katt_Dwx": 0.00004,
    #         "ctrlSJC/Katt_Dwy": 0.00004,
    #         "ctrlSJC/Katt_Dwz": 0.00001,
    #         # attitude I
    #         "ctrlSJC/Katt_Ix": 0.002,
    #         "ctrlSJC/Katt_Iy": 0.002, 
    #         "ctrlSJC/Katt_Iz": 0.004,
    #         "ctrlSJC/Katt_I_limit": 0,
    #     }
    # },

    # {
    #     "name": "S2S",
    #     "heights": [0.5, 0.8],
    #     "ids": [50, 51],
    #     "firmware_params":
    #     {
    #         "motorPowerSet/mixMode": 1,
    #         # attitude lambda
    #         "ctrlSJC/Katt_Px": 8,
    #         "ctrlSJC/Katt_Py": 8,
    #         "ctrlSJC/Katt_Pz": 4,
    #         # attitude K
    #         "ctrlSJC/Katt_Dx": 0.0005,
    #         "ctrlSJC/Katt_Dy": 0.0005,
    #         "ctrlSJC/Katt_Dz": 0.0005,
    #         # delay
    #         "ctrlSJC/Katt_Dwx": 0.00004,
    #         "ctrlSJC/Katt_Dwy": 0.00004,
    #         "ctrlSJC/Katt_Dwz": 0.00001,
    #         # attitude I
    #         "ctrlSJC/Katt_Ix": 0.002,
    #         "ctrlSJC/Katt_Iy": 0.002, 
    #         "ctrlSJC/Katt_Iz": 0.004,
    #         "ctrlSJC/Katt_I_limit": 0,
    #     }
    # },

    # {
    #     "name": "S2S",
    #     "heights": [0.5, 0.8],
    #     "ids": [50, 51],
    #     "firmware_params":
    #     {
    #         "motorPowerSet/mixMode": 1,
    #         # attitude lambda
    #         "ctrlSJC/Katt_Px": 8,
    #         "ctrlSJC/Katt_Py": 8,
    #         "ctrlSJC/Katt_Pz": 4,
    #         # attitude K
    #         "ctrlSJC/Katt_Dx": 0.001,
    #         "ctrlSJC/Katt_Dy": 0.001,
    #         "ctrlSJC/Katt_Dz": 0.001,
    #         # delay
    #         "ctrlSJC/Katt_Dwx": 0.00008,
    #         "ctrlSJC/Katt_Dwy": 0.00008,
    #         "ctrlSJC/Katt_Dwz": 0.00002,
    #         # attitude I
    #         "ctrlSJC/Katt_Ix": 0.002,
    #         "ctrlSJC/Katt_Iy": 0.002, 
    #         "ctrlSJC/Katt_Iz": 0.004,
    #         "ctrlSJC/Katt_I_limit": 0,
    #     }
    # },

    # {
    #     "name": "S2S",
    #     "heights": [0.5, 0.8],
    #     "ids": [50, 51],
    #     "firmware_params":
    #     {
    #         "motorPowerSet/mixMode": 1,
    #         # attitude lambda
    #         "ctrlSJC/Katt_Px": 8,
    #         "ctrlSJC/Katt_Py": 8,
    #         "ctrlSJC/Katt_Pz": 4,
    #         # attitude K
    #         "ctrlSJC/Katt_Dx": 0.001,
    #         "ctrlSJC/Katt_Dy": 0.001,
    #         "ctrlSJC/Katt_Dz": 0.001,
    #         # delay
    #         "ctrlSJC/Katt_Dwx": 0.0001,
    #         "ctrlSJC/Katt_Dwy": 0.0001,
    #         "ctrlSJC/Katt_Dwz": 0.00005,
    #         # attitude I
    #         "ctrlSJC/Katt_Ix": 0.002,
    #         "ctrlSJC/Katt_Iy": 0.002, 
    #         "ctrlSJC/Katt_Iz": 0.004,
    #         "ctrlSJC/Katt_I_limit": 0,
    #     }
    # },

    {
        "name": "L2L",
        "heights": [0.5, 0.8],
        "ids": [101, 102],
        "firmware_params":
        {
            "motorPowerSet/mixMode": 2,
        #     # attitude lambda
        #     "ctrlSJC/Katt_Px": 10
        #     "ctrlSJC/Katt_Py": 10,
            # "ctrlSJC/Katt_Pz": 5 * 0.2,
        #     # attitude K
        #     "ctrlSJC/Katt_Dx": 0.01,
        #     "ctrlSJC/Katt_Dy": 0.01,
            # "ctrlSJC/Katt_Dz": 0.0025 * 0.2,
        #     # delay
        #     "ctrlSJC/Katt_Dwx": 0.0002,
        #     "ctrlSJC/Katt_Dwy": 0.0002,
        #     "ctrlSJC/Katt_Dwz": 0.0, #0.00005,
        #     "ctrlSJC/Katt_Dw_limit": 10000,
        #     # attitude I
        #     "ctrlSJC/Katt_Ix": 0.002,
        #     "ctrlSJC/Katt_Iy": 0.002, 
            # "ctrlSJC/Katt_Iz": 0.004 * 0.0,
        #     "ctrlSJC/Katt_I_limit": 2,

        #     # position controller
        #     "ctrlSJC/Kpos_Px": 20,
        #     "ctrlSJC/Kpos_Py": 20,
        #     "ctrlSJC/Kpos_Pz": 10,

        #     "ctrlSJC/Kpos_Dx": 10,
        #     "ctrlSJC/Kpos_Dy": 10,
        #     "ctrlSJC/Kpos_Dz": 5,

        #     "ctrlSJC/Kpos_Ix": 3,
        #     "ctrlSJC/Kpos_Iy": 3,
        #     "ctrlSJC/Kpos_Iz": 10,

        #     "ctrlSJC2/T_d_lambda": 10,
        #     "ctrlSJC2/T_d_dot_limit": 1,
        }
    },


]

SwapTimes = [4]
Radius = 0.25

if __name__ == "__main__":

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # enable ground effect in the NN computation
    allcfs.setParam("ctrlFa/enableGround", 1)
    allcfs.setParam("ctrlFa/filter", 0.35)
    allcfs.setParam("ctrlFa/enableNN", 2)


    for config in configs:
        print("Executing: ", config["name"])
        Ids = config["ids"]

        if "firmware_params" in config:
            for param, value in config["firmware_params"].items():
                allcfs.setParam(param, value)

        Heights = config["heights"]

        # computing starting point angles
        angles = np.linspace(np.pi/2, 2.5*np.pi, len(Ids)+1, endpoint=False)[0:len(Ids)]
        angles = np.array([0, np.pi/2])

        allcfs.takeoff(targetHeight=np.min(Heights), duration=3.0)
        timeHelper.sleep(3.5)

        # go to initial positions
        for angle, cfid, height in zip(angles[0:len(Ids)], Ids, Heights):
            pos = np.array([np.cos(angle) * Radius, np.sin(angle) * Radius, height])
            allcfs.crazyfliesById[cfid].goTo(pos, 0, 3.0)

        timeHelper.sleep(3.5)

        allcfs.setParam("usd/logging", 1)

        for swapTime in SwapTimes:
            # swap 1
            for angle, cfid, height in zip(angles, Ids, Heights):
                pos = np.array([np.cos(angle + np.pi) * Radius, np.sin(angle + np.pi) * Radius, height])
                allcfs.crazyfliesById[cfid].goTo(pos, 0, swapTime)
            timeHelper.sleep(swapTime)

            # swap 2
            for angle, cfid, height in zip(angles, Ids, Heights):
                pos = np.array([np.cos(angle) * Radius, np.sin(angle) * Radius, height])
                allcfs.crazyfliesById[cfid].goTo(pos, 0, swapTime)
            timeHelper.sleep(swapTime)

        allcfs.setParam("usd/logging", 0)

        for cfid in Ids:
            cf = allcfs.crazyfliesById[cfid]
            pos = np.array(cf.initialPosition) + np.array([0, 0, np.min(Heights)])
            cf.goTo(pos, 0, 3.0)
        timeHelper.sleep(3.5)

        # land flying CFs
        allcfs.land(targetHeight=0.02, duration=3.0)
        timeHelper.sleep(5.0)
