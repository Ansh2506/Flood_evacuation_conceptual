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
from SALib.test_functions import Ishigami
import numpy as np
import csv
from agent import Human
from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
import os
from os import listdir, path
floor_plans = [
    f
    for f in listdir("floorplans")
    if path.isfile(path.join("floorplans", f))
]

sensitivity_folder= os.path.dirname(os.path.dirname(os.path.realpath(__file__))) +"\outputs_for_sensitivity_analysis"

 
print(sensitivity_folder)


header = ['human_count', 'collaboration_percentage', 'flood_probability', 'route_information_percentage', 'believes_alarm','mobility_good_percentage','dead',]
with open(sensitivity_folder+'/output'+timestamp+'.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    f.close()

# Define the model inputs
def sensitivity_parameters():
    problem = {
        'num_vars': 6,
        'names': ['human_count', 'collaboration_percentage', 'flood_probability', 'route_information_percentage', 'believes_alarm','mobility_good_percentage'],
        'bounds': [[1, 10],[0, 100],[0, 100],[0,100],[0,0.9],[0,100]]
        }

    distinct_samples = 64  #one sample results in 2^N different combinations: We have N variables, and we take 2 distinct values per sample per variable

    # We get all our samples for the SA here

    param_values = saltelli.sample(problem, distinct_samples)
    return param_values

def RunModel():
    problem = {
        'num_vars': 6,
        'names': ['human_count', 'collaboration_percentage', 'flood_probability', 'route_information_percentage', 'believes_alarm','mobility_good_percentage'],
        'bounds': [[1, 10],[0, 100],[0, 100],[0,100],[0,0.9],[0,100]]
        }
    param_values = sensitivity_parameters()
    batch = len(sensitivity_parameters())

    # We define our variables and bounds
    
    
    y = np.zeros([param_values.shape[0]])
    for i in range(0,batch):
        print("sample_no", i)
        iterations_for_one_sample=10
        
        l=[]
        for j in range(0,iterations_for_one_sample):
            print("iterations_no",j)
            from model import FloodEvacuation
            FloodEvacuation=FloodEvacuation(floor_plan_file=floor_plans[5],
                            human_count=int(param_values[i][0]),
                            collaboration_percentage= float(param_values[i][1]),
                            flood_probability=float(param_values[i][2]),
                            mobility_good_percentage= float(param_values[i][5]),
                            random_spawn=False,
                            visualise_vision=False,
                            save_plots=True,
                            route_information_percentage=float(param_values[i][3]))

            FloodEvacuation.believes_alarm = float(param_values[i][4])

            print("human_count",int(param_values[i][0]))
            print("collaboration_percentage",float(param_values[i][1]))
            print("prob_flood",float(param_values[i][2]))
            print("route_info",float(param_values[i][3]))
            print("believes_alarm",float(param_values[i][4]))
            print("mobility_per",float(param_values[i][5]))

            FloodEvacuation.run_model(num_steps=50)
            dead_persons=FloodEvacuation.count_human_status(FloodEvacuation,Human.Status.DEAD)
            l.append(dead_persons)
        avg_value_of_dead_persons=sum(l)/len(l)
        y[i]=int(avg_value_of_dead_persons)
        data= [str(FloodEvacuation.human_count),str(FloodEvacuation.collaboration_percentage), str(FloodEvacuation.flood_probability),str(FloodEvacuation.route_information_percentage),str(FloodEvacuation.believes_alarm),str(FloodEvacuation.mobility_good_percentage),str(int(avg_value_of_dead_persons))]
        with open(sensitivity_folder+'/output'+timestamp+'.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
            f.close()

    Si = sobol.analyze(problem, y, print_to_console=True)

    print(Si['S1'])


RunModel()


