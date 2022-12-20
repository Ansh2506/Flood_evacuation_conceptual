import os
import numpy as np
import pandas as pd
import yaml
import argparse 
import datetime
import json
import SALib
import datetime as dt

from SALib.sample import saltelli
from SALib.analyze import sobol

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np


#from run import get_study_area #i do not have a designated study area
from model import FloodEvacuation
import run

#set globals
#ROOT =r'/scistor/ivm/isk520/polygene/models/down2earth' #what is this ?

#OUTPUT =r'/scistor/ivm/isk520/polygene/models/down2earth/DataDrive/Sub-Ewaso/Output' 
OUTPUT = r'c:/Users/z5239100/OneDrive - UNSW/PhD/Research Visit/Codes/basicfloodevac-master_28_11_2022/basicfloodevac-master/flood/Output'

'''again not sure what these are, perhaps not applicable to me'''



# Define the model inputs

def sensitivity_parameters():
    problem = {
        'num_vars': 5,
        'names': ['human_count', 'collaboration_percentage', 'flood_probability', 'route_information_percentage', 'believes_alarm'],
        'bounds': [[1, 100],[0, 100],[0, 100],[0,100],[0,0.9]]
        }

    distinct_samples = 32  #one sample results in 2^N different combinations: We have N variables, and we take 2 distinct values per sample per variable

    # We get all our samples for the SA here

    param_values = saltelli.sample(problem, distinct_samples)
    return param_values

# Run model (example)
def RunModel():
    # Loop through each parameter combination? For now test with 100 combinations
    # Set globals
    # Set the repetitions, the amount of steps, and the amount of distinct values per variable
    run = 0
    replicates = 1 # you repeat because of stochastic initial conditions

    param_values = sensitivity_parameters()
    sa_results_dict = False
    batch = len(sensitivity_parameters())

    # We define our variables and bounds
    
    for i in range(0, batch):

        # Create folder to store run results and settings file
        run_id = 'output1_'+ str(i)
        f_run = os.path.join(OUTPUT, run_id)
        if not os.path.exists(f_run):
            os.mkdir(f_run)
        
        # # Save model settings (now stored in config) '''what is going on here'''
        FloodEvacuation.human_count = float(param_values[i][0])
        FloodEvacuation.collaboration_percentage = float(param_values[i][1])
        FloodEvacuation.flood_probability = float(param_values[i][2])
        FloodEvacuation.route_information_percentage= float(param_values[i][3])
        FloodEvacuation.believes_alarm = float(param_values[i][4])

        # = os.path.join(f_run, 'report_'+str(i)) # change directory output
        
        run()
        #store al output

RunModel()
# Perform analysis
#Si = sobol.analyze(problem, Y, print_to_console=True)

# Print the first-order sensitivity indices
#print(Si['S1'])
