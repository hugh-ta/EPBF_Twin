import pyPenred
from pathlib import Path
import os

os.chdir(Path(__file__).resolve().parent)

config = pyPenred.readConfigFile("config.in") # Load the file as a dictionary
config['simulation']['threads'] = 32  # Modify parameters
config['sources']['generic']['source1']['nhist'] = 1.0e7
pyPenred.writeConfigFile("modified.in", config)

# running the sim
pyPenred.runFromFile("modified.in")

#yay it works

