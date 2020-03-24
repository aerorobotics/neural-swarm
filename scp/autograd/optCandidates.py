import numpy as np
import argparse
import subprocess
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import os, shutil, glob
import datetime
import yaml
import itertools
from scp import scp
from robots import RobotDoubleIntegrator, RobotDubinsCar, RobotAirplane


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  args = parser.parse_args()

  with open("dubinsCar.yaml", 'r') as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

  # robot = RobotDoubleIntegrator()
  robot = RobotDubinsCar(v=data["robot"]["v"], k=data["robot"]["k"])
  # robot = RobotAirplane()

  data = np.loadtxt("../../buildDebug/result.csv",delimiter=' ')

  scp(robot, initialU = data[:,3:4], initialX = data[:,0:3], dt = 0.01, goalPos = [55,45], pdfFile = "optCand.pdf")