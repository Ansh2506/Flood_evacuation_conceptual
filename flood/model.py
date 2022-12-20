from math import floor
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
from datetime import datetime, timedelta

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import Coordinate, MultiGrid
from mesa.time import RandomActivation

from agent import Human, Wall ,EmergencyExit, House, Water, Tree, Bridge,Tile, Highway, Path, AgricultureField,Forecaster


class FloodEvacuation(Model):
    MIN_HEALTH = 4
    MAX_HEALTH = 7

    MIN_SPEED = 0.0
    MAX_SPEED = 8.0

    MIN_NERVOUSNESS = 1
    MAX_NERVOUSNESS = 10

    MIN_EXPERIENCE = 1
    MAX_EXPERIENCE = 10

    MIN_VISION = 1
    MAX_VISION = 10
    # MAX_VISION is simply the size of the grid

    def __init__(
        self,
        floor_plan_file: str,
        human_count: int,
        collaboration_percentage: float,
        route_information_percentage: float,
        flood_probability: float,
        mobility_good_percentage: float,
        visualise_vision: bool,
        random_spawn: bool,
        save_plots: bool,
        #believes_alarm : bool,
    ):
        # Load floorplan
        # floorplan = np.genfromtxt(path.join("flood/floorplans/", floor_plan_file))
        with open(os.path.join("floorplans/", floor_plan_file), "rt") as f:
            floorplan = np.matrix([line.strip().split() for line in f.readlines()])

        # Rotate the floorplan so it's interpreted as seen in the text file
        floorplan = np.rot90(floorplan, 3)

        # Check what dimension our floorplan is
        width, height = np.shape(floorplan)

        # Init params
        self.width = width
        self.height = height
        self.human_count = human_count
        self.collaboration_percentage = collaboration_percentage
        self.route_information_percentage = route_information_percentage
        self.mobility_good_percentage = mobility_good_percentage
        self.visualise_vision = visualise_vision
        self.flood_probability = flood_probability
        self.flood_started = False  # Turns to true when a flood has started
        self.save_plots = save_plots

        # Set up model objects
        self.schedule = RandomActivation(self)
        
        self.start_time = datetime(2022, 10, 17, 0, 0, 0)
        self.curr_time = self.start_time
        self.time_step = 0
        self.timestep_length = timedelta(minutes=30, hours=0)

        self.grid = MultiGrid(height, width, torus=False)
        
        #alarm attributes
        self.alarm = False

        # Used to start a flood at a random house location
        self.house: dict[Coordinate, House] = {}
        self.trees: dict[Coordinate, Tree] = {}
        self.bridges: dict[Coordinate, Bridge] = {}
        self.highway : dict[Coordinate, Highway] = {}
        self.path : dict[Coordinate, Path] = {}
        self.agriculture_field : dict[Coordinate, AgricultureField] = {}

        # Used to easily see if a location is a shelter, since this needs to be done a lot
        self.flood_exits: dict[Coordinate, EmergencyExit] = {}
       
        # If random spawn is false, spawn_pos_list will contain the list of possible spawn points according to the floorplan
        self.random_spawn = random_spawn
        self.spawn_pos_list: list[Coordinate] = []

        self.river_water: dict[Coordinate,Water] = {}

        # Load floorplan objects
        for (x, y), value in np.ndenumerate(floorplan):
            pos: Coordinate = (x, y)

            # e = Elevation(self,pos,pos[0]*pos[1])
            # self.grid.place_agent(e,pos)
            value = str(value)
            floor_object = None
            if value == "W":
                floor_object = Wall(pos, self)
            elif value == "E":
                floor_object = EmergencyExit(pos, self)
                #print()
                self.flood_exits[pos] = floor_object
                
            elif value == "B":
                floor_object = House(pos, self)
                self.house[pos] = floor_object
            elif value == "T":
                floor_object = Tree(pos,self)
                self.trees[pos] = floor_object
            elif value == "b": 
                floor_object = Bridge(pos, self)
                self.bridges[pos] = floor_object
            elif value == "H":
                floor_object = Highway(pos, self)
                self.highway[pos] = floor_object
            elif value == "P":
                floor_object = Path(pos, self)
                self.path[pos] = floor_object
            elif value == "A":
                floor_object = AgricultureField(pos, self)
                self.agriculture_field[pos] = floor_object
            elif value == "S":
                self.spawn_pos_list.append(pos)
            elif value == "R":
                floor_object = Water(pos, self)
                self.river_water[pos] = floor_object
                self.flood_started = True
                floor_object2 = Tile(pos,6,self)
                self.grid.place_agent(floor_object2,pos)
                self.schedule.add(floor_object2)
            
            elif value == "_":
                #num = random.randint(0,6)
                num = 0
                floor_object = Tile(pos,num,self)

            if floor_object:
                self.grid.place_agent(floor_object, pos)
                self.schedule.add(floor_object)
                
        assert len(self.spawn_pos_list) > 0
          #elevation layer will be added as space
         #self.space.set_elevation_layer("data/elevation.asc.gz", crs="epsg:4326")
        # Create a graph of traversable routes, used by agents for pathing
        self.graph = nx.Graph()
        for agents, x, y in self.grid.coord_iter():
            pos = (x, y)

            # If the location is empty, or there are no non-traversable agents
            if len(agents) == 0 or not any(not agent.traversable for agent in agents):
                neighbors_pos = self.grid.get_neighborhood(
                    pos, moore=True, include_center=True, radius=1
                )

                for neighbor_pos in neighbors_pos:
                    # If the neighbour position is empty, or no non-traversable contents, add an edge
                    if self.grid.is_cell_empty(neighbor_pos) or not any(
                        not agent.traversable
                        for agent in self.grid.get_cell_list_contents(neighbor_pos)
                    ):
                        self.graph.add_edge(pos, neighbor_pos)

        # Collects statistics from our model run
        self.datacollector = DataCollector(
            {
                "Alive": lambda m: self.count_human_status(m, Human.Status.ALIVE),
                "Dead": lambda m: self.count_human_status(m, Human.Status.DEAD),
                "Escaped": lambda m: self.count_human_status(m, Human.Status.ESCAPED),
                
                "Normal": lambda m: self.count_human_mobility(m, Human.Mobility.NORMAL),
                "Erratic": lambda m: self.count_human_mobility(m, Human.Mobility.ERRATIC),
                "Verbal Collaboration": lambda m: self.count_human_collaboration(
                    m, Human.Action.VERBAL_SUPPORT
                ),
                
            }
        )

        # Calculate how many agents will be collaborators
        number_collaborators = int(round(self.human_count * (self.collaboration_percentage / 100)))
        number_of_route_information = int(round(self.human_count * (self.route_information_percentage / 100)))
        number_of_good_mobility = int(round(self.human_count * (self.mobility_good_percentage / 100)))
        print(number_collaborators)
        print(number_of_route_information)
        print("number with good mobility:", number_of_good_mobility)

        
        # Start placing human agents
        for i in range(0, self.human_count):     
            if self.random_spawn:  # Place human agents randomly
                pos = self.grid.find_empty()
            else:  # Place human agents at specified spawn locations
                n = len(self.spawn_pos_list)
                print(n)
                pos = self.spawn_pos_list[random.randint(0,n-1)]

            if pos:
                #Create a random human
                #health = np.random.randint(self.MIN_HEALTH * 100, self.MAX_HEALTH * 100) / 100
                health = np.random.randint(self.MIN_HEALTH,self.MAX_HEALTH)
                #speed = np.random.randint(self.MIN_SPEED, self.MAX_SPEED)
                #speed = self.MIN_SPEED
                speed = 1

                if number_collaborators > 0:
                    collaborates = True
                    number_collaborators -= 1
                else:
                    collaborates = False

                if number_of_route_information > 0:
                    route_information = True
                    number_of_route_information -=1
                    print(number_of_route_information)
                else:
                    route_information = False

                if number_of_good_mobility > 0:
                    mobility_good = True
                    number_of_good_mobility -=1
                    print("number with good mobility:",number_of_good_mobility)
                else:
                    mobility_good = False

                # Vision statistics obtained from http://www.who.int/blindness/GLOBALDATAFINALforweb.pdf
                
                vision_distribution = [0.0058, 0.0365, 0.0424, 0.9153]
                vision = int(
                    np.random.choice(
                        np.arange(
                            self.MIN_VISION,
                            self.width + 1,
                            (self.width / len(vision_distribution)),
                        ),
                        p=vision_distribution,
                    )
                )

                nervousness_distribution = [
                    0.025,
                    0.025,
                    0.1,
                    0.1,
                    0.1,
                    0.3,
                    0.2,
                    0.1,
                    0.025,
                    0.025,
                ]  # Distribution with slight higher weighting for above median nerovusness
                nervousness = int(
                    np.random.choice(
                        range(self.MIN_NERVOUSNESS, self.MAX_NERVOUSNESS + 1),
                        p=nervousness_distribution,
                    )
                )  # Random choice starting at 1 and up to and including 10

                experience = np.random.randint(self.MIN_EXPERIENCE, self.MAX_EXPERIENCE)

                belief_distribution = [0.9, 0.1]  # [Believes, Doesn't Believe]
                believes_alarm = np.random.choice([True, False], p=belief_distribution) #p is the probability of each element in array
                #believes_alarm = 0.2 * self.human_count
                print("Does the agent believe the alarm :", believes_alarm)

                self_warned = False 
                age = np.random.randint(20, 70) 

                xyz_distribution = [0.6,0.4]
                xyz = np.random.choice([True, False], p=xyz_distribution)  

                human = Human(
                    pos,
                    health=health,
                    speed=speed,
                    vision=vision,
                    collaborates=collaborates,
                    route_information= route_information,
                    mobility_good= mobility_good,
                    nervousness=nervousness,
                    experience=experience,
                    believes_alarm=believes_alarm,
                    self_warned = self_warned,
                    age = age,
                    model=self,
                )
                print(human.vision)
                self.grid.place_agent(human, pos)
                self.schedule.add(human)
            else:
                print("No tile empty for human placement!")
                
        self.initial_risk = 0.5
        forecaster_agent = Forecaster(
        self.initial_risk,
        pos,
        #activates = False,
        #pos,
        model =self,
        #model = self)
        activates = False)
        self.schedule.add(forecaster_agent)
        
        self.running = True

    # Plots line charts of various statistics from a run
    def save_figures(self):
        DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        OUTPUT_DIR = DIR + "/output"

        results = self.datacollector.get_model_vars_dataframe()

        dpi = 100
        fig, axes = plt.subplots(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi, nrows=1, ncols=3)

        status_results = results.loc[:, ["Alive", "Dead", "Escaped"]]
        status_plot = status_results.plot(ax=axes[0])
        status_plot.set_title("Human Status")
        status_plot.set_xlabel("Simulation Step")
        status_plot.set_ylabel("Count")

        mobility_results = results.loc[:, ["Normal", "Erratic"]]
        mobility_plot = mobility_results.plot(ax=axes[1])
        mobility_plot.set_title("Human Mobility")
        mobility_plot.set_xlabel("Simulation Step")
        mobility_plot.set_ylabel("Count")

        collaboration_results = results.loc[
            :, ["Verbal Collaboration"]
        ]
        collaboration_plot = collaboration_results.plot(ax=axes[2])
        collaboration_plot.set_title("Human Collaboration")
        collaboration_plot.set_xlabel("Simulation Step")
        collaboration_plot.set_ylabel("Successful Attempts")
        collaboration_plot.set_ylim(ymin=0)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt.suptitle(
            "Percentage Collaborating: "
            + str(self.collaboration_percentage)
            + "%, Number of Human Agents: "
            + str(self.human_count),
            fontsize=16,
        )
        plt.savefig(OUTPUT_DIR + "/model_graphs/" + timestr + ".png")
        plt.close(fig)

    

    def step(self):
        """
        Advance the model by one step.
        """

        self.schedule.step()
        #self.current_time = self.current_time + self.timestep_length
        self.time_step = self.time_step + 1
        self.curr_time = self.start_time + self.schedule.steps * self.timestep_length
        print(self.time_step)
        #print(number_of_route_information)
        

        

        self.datacollector.collect(self)

        # If no more agents are alive, stop the model and collect the results
        if self.count_human_status(self, Human.Status.ALIVE) == 0:
            self.running = False

            if self.save_plots:
                self.save_figures()

    @staticmethod
    def count_human_collaboration(model, collaboration_type):
        """
        Helper method to count the number of collaborations performed by Human agents in the model
        """

        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human):
                if collaboration_type == Human.Action.VERBAL_SUPPORT:
                    count += agent.get_verbal_collaboration_count()
                #elif collaboration_type == Human.Action.MORALE_SUPPORT:
                    #count += agent.get_morale_collaboration_count()
                

        return count

    @staticmethod
    def count_human_status(model, status):
        """
        Helper method to count the status of Human agents in the model
        """
        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human) and agent.get_status() == status:
                count += 1

        return count

    @staticmethod
    def count_human_mobility(model, mobility):
        """
        Helper method to count the mobility of Human agents in the model
        """
        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human) and agent.get_mobility() == mobility:
                count += 1

        return count
