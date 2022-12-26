from typing import Union
from typing_extensions import Self
from mesa.space import Coordinate
import networkx as nx
import numpy as np
from enum import IntEnum
from mesa import Agent
from copy import deepcopy
from utils import get_random_id
import random


def get_line(start, end):
    """
    Implementation of Bresenham's Line Algorithm
    Returns a list of tuple coordinates from starting tuple to end tuple (and including them)
    """
    # Break down start and end tuples
    x1, y1 = start
    x2, y2 = end

    # Calculate differences
    diff_x = x2 - x1
    diff_y = y2 - y1

    # Check if the line is steep
    line_is_steep = abs(diff_y) > abs(diff_x)

    # If the line is steep, rotate it
    if line_is_steep:
        # Swap x and y values for each pair
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # If the start point is further along the x-axis than the end point, swap start and end
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Calculate the differences again
    diff_x = x2 - x1
    diff_y = y2 - y1

    # Calculate the error margin
    error_margin = int(diff_x / 2.0)
    step_y = 1 if y1 < y2 else -1

    # Iterate over the bounding box, generating coordinates between the start and end coordinates
    y = y1
    path = []

    for x in range(x1, x2 + 1):
        # Get a coordinate according to if x and y values were swapped
        coord = (y, x) if line_is_steep else (x, y)
        path.append(coord)  # Add it to our path
        # Deduct the absolute difference of y values from our error_margin
        error_margin -= abs(diff_y)

        # When the error margin drops below zero, increase y by the step and the error_margin by the x difference
        if error_margin < 0:
            y += step_y
            error_margin += diff_x 

    # The the start and end were swapped, reverse the path
    if swapped:
        path.reverse()

    return path


"""
FLOOR OBJECTS
"""
class FloorObject(Agent):
    def __init__(
        self,
        pos: Coordinate,
        traversable: bool,
        model = None,
    ):
        rand_id = get_random_id()
        super().__init__(rand_id,model)
        self.pos = pos
        self.traversable = traversable
        
    def get_position(self):
        return self.pos

class Sight(FloorObject):
    def __init__(self, pos, model):
        super().__init__(
            pos, traversable=True,model=model
        )

    def get_position(self):
        return self.pos


class EmergencyExit(FloorObject):
    def __init__(self, pos, model):
        super().__init__(
            pos, traversable=True,model=model
        )

class Wall(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=False, model=model)

class House(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=False, model=model)

class Tree(FloorObject):
    def __init__(self,pos,model):
        super().__init__(pos,traversable=False,model=model)

class Bridge(FloorObject):
    def __init__(self,pos,model):
        super().__init__(pos,traversable=True,model=model)

class DeadHuman(FloorObject):
    def __init__(self,pos,model):
        super().__init__(pos,traversable=True,model=model)

    def get_status(self):
        return Human.Status.DEAD

class Tile(FloorObject):
    def __init__(self,pos,elevation,model):
        super().__init__(pos,traversable=True,model=model)
        self.elevation = elevation
        
class Highway(FloorObject):
    def __init__(self,pos,model):
        super().__init__(pos,traversable=True,model=model)
        
class Path(FloorObject):
    def __init__(self,pos,model):
        super().__init__(pos,traversable=True,model=model)
        
class AgricultureField(FloorObject):
    def __init__(self,pos,model):
        super().__init__(pos,traversable=True,model=model)



"""
WATER STUFF
"""

class Water(FloorObject):  

    def __init__(self,pos,model):
        super().__init__(
            pos,
            traversable=True,
            model=model,
        )
        self.elevation = 21
        self.speed = 1       #this can be changed to our wish,and affects the water radius  

    def step(self):
        dont_spread = [Water,House,Wall,Bridge]
        neighborhood = self.model.grid.get_neighborhood(self.pos,moore=True,include_center=False,radius=self.speed)
        own_contents = self.model.grid.get_cell_list_contents(self.pos)
        own_elevation = 0 
        
        
        for agent in own_contents:
            if isinstance(agent,Tile):
                own_elevation = agent.elevation
                
        for cell in neighborhood:
            cont = self.model.grid.get_cell_list_contents((cell[0],cell[1])) #x and y coordinates of the neighbouring cells, get contents of neighbouring cells
            flag = False
            for agent in cont:
                for i in dont_spread:
                    if isinstance(agent,i):
                        flag = True
                        break
                    
            if flag == False:
                neighbour_elevation = 0
                for agent in cont:
                    if isinstance(agent,Tile):
                        neighbour_elevation = agent.elevation
                        
                if neighbour_elevation <= own_elevation:  
                        
                    water = Water((cell[0],cell[1]),self.model)
                    water.unique_id = get_random_id()
                    self.model.schedule.add(water)
                    self.model.grid.place_agent(water,(cell[0],cell[1]))

    def get_position(self):
        return super().get_position()

class Government(Agent):
    def __init__(self,
                 #unique_id,
                 #initial_risk,
                activates: bool,
                pos: Coordinate,
                #response,
                model):
        self.model = model
        rand_id = get_random_id()
        super().__init__(rand_id, model)
        
        
        self.activates = activates
        self.pos = pos
        #self.vision = vision
        #self.planned_response : tuple[Agent,Coordinate] = (None,None,)
        #self.visible_area : tuple[Coordinate, tuple[Agent]] = []
        
    def get_position(self):
        return self.pos
        
 
class Forecaster(Government):
    def __init__(self, initial_risk,
                 #activation_threshold,
                pos,activates,model):
        super().__init__(activates,pos,model)
        
        self.initial_risk = initial_risk
        #self.activation_threshold = activation_threshold
        #what the agent does at each step
    def step(self):
        #activation only when risk is assessed
        #if random.uniform(0,1) > self.initial_risk: #if number greater than initial risk then agent gets activated
            #self.activates = True
        
        #activation only certain timesteps
        #if self.model.time_step == 1 or 3 or 5 or 7 or 9 :
            #return

        if self.model.time_step == 1:
            print("forecaster agent activated")
            self.activates = True

        if self.activates == True: ##and the threshold is a certain value but action isn't taken 
            self.take_action()
        
                   
    def take_action(self):
        # for agents in self.visible_tiles:
        #     for agent in agents:
        #         if isinstance(agent,Human):
        if self.model.alarm == False:
                    self.model.alarm = True 
                    print('Forecast released')#how will this alarm be released, maybe only available as information to agents
        # add threshold to take action or do nothing, low and high threshold 
                    
                        
   

class Human(Agent):
    """
    A human agent, which will attempt to escape from the grid.

    Attributes:
        ID: Unique identifier of the Agent
        Position (x,y): Position of the agent on the Grid
        Health: Health of the agent (between 0 and 1)
        ...
    """
    class Mobility(IntEnum):
        NORMAL = 1

    class Status(IntEnum):
        DEAD = 0
        ALIVE = 1
        ESCAPED = 2

    class Action(IntEnum):
        MORALE_SUPPORT = 1
        VERBAL_SUPPORT = 2

    class Awareness(IntEnum):
        UNAWARE = 0
        AWARE = 1


    MIN_HEALTH = 0.0
    MAX_HEALTH = 5.0

    MIN_EXPERIENCE = 1
    MAX_EXPERIENCE = 10

    MIN_SPEED = 0.0

    MIN_KNOWLEDGE = 0
    MAX_KNOWLEDGE = 1
    # When the health value drops below this value, the agent will being to slow down
    SLOWDOWN_THRESHOLD = 0.5

    def __init__(
        self,
        pos: Coordinate,
        health: float,
        speed: float,
        collaborates: bool,
        route_information : bool,
        vision,
        experience,
        believes_alarm: bool,
        self_warned: bool,
        age,
        mobility_good,
        model,
    ):
        rand_id = get_random_id()
        super().__init__(rand_id, model)

        print(self.unique_id)
        
        self.traversable = False
        self.pos = pos
        self.health = health
        self.mobility: Human.Mobility = Human.Mobility.NORMAL
        self.mobility_good = mobility_good
        self.speed = speed
        self.vision = vision
        self.collaborates = collaborates
        self.route_information = route_information
        self.awareness : Human.Awareness = Human.Awareness.UNAWARE

        self.verbal_collaboration_count: int = 0
     
        self.morale_boost: bool = False
        
        self.knowledge = self.MIN_KNOWLEDGE
        self.experience = experience
        self.believes_alarm = believes_alarm
        self.self_warned = self_warned
        self.escaped: bool = False

        self.planned_target: tuple[Agent,Coordinate] = (
            None,
            None,
        )

        self.planned_action: Human.Action = None

        self.visible_tiles : tuple[Coordinate, tuple[Agent]] = []    #Think how this will play out in water context

        self.known_tiles: dict[Coordinate, set[Agent]] = {}

        self.visited_tiles: set[Coordinate] = {self.pos}

    ##why is this done ? 
    def update_sight_tiles(self, visible_neighborhood):
        if len(self.visible_tiles) > 0:
            # Remove old vision tiles
            for pos, _ in self.visible_tiles:
                contents = self.model.grid.get_cell_list_contents(pos)
                for agent in contents:
                    if isinstance(agent, Sight):
                        self.model.grid.remove_agent(agent)
                        print("sight removed")

        # Add new vision tiles
        for contents, tile in visible_neighborhood:
            # Don't place if the tile has contents but the agent can't see it
            if self.model.grid.is_cell_empty(tile) or len(contents) > 0:
                sight_object = Sight(tile, self.model)
                self.model.grid.place_agent(sight_object, tile)

    # A strange implementation of ray-casting, using Bresenham's Line Algorithm, which takes into account 
    #smoke and visibility of objects
    
    def get_visible_tiles(self) -> tuple[Coordinate, tuple[Agent]]:
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=self.vision
        )
        visible_neighborhood = set()

        # A set of already checked tiles, for avoiding repetition and thus increased efficiency
        checked_tiles = set()

        # Reverse the neighborhood so we start from the furthest locations and work our way inwards
        for pos in reversed(neighborhood):
            if pos not in checked_tiles:
                blocked = False
                try:
                    path = get_line(self.pos, pos)

                    for i, tile in enumerate(path):
                        contents = self.model.grid.get_cell_list_contents(tile)
                        visible_contents = []
                        for obj in contents:
                            if isinstance(obj, Sight):
                                # ignore sight tiles
                                continue
                            elif isinstance(obj, Wall):
                                # We hit a wall, reject rest of path and move to next
                                blocked = True
                                break
                            if not isinstance(obj,Sight):
                                visible_contents.append(obj)

                        if blocked:
                            checked_tiles.update(
                                path[i:]
                            )  # Add the rest of the path to checked tiles, since we now know they are not visible
                            break
                        else:
                            # If a wall didn't block the way, add the visible agents at this location
                            checked_tiles.add(
                                tile
                            )  # Add the tile to checked tiles so we don't check it again
                            visible_neighborhood.add((tile, tuple(visible_contents)))

                except Exception as e:
                    print(e)

        if self.model.visualise_vision:
            self.update_sight_tiles(visible_neighborhood)

        return tuple(visible_neighborhood)


    def get_random_target(self, allow_visited=True):
        graph_nodes = self.model.graph.nodes() #perform set like operations, finds shortest path 

        known_pos = set(self.known_tiles.keys())

        #If we are excluding visited tiles, remove the visited_tiles set from the available tiles
        if not allow_visited:
            known_pos -= self.visited_tiles

        traversable_pos = [pos for pos in known_pos if self.location_is_traversable(pos)]

        while not self.planned_target[1]:
            i = np.random.choice(len(traversable_pos))
            target_pos = traversable_pos[i]
            if target_pos in graph_nodes and target_pos != self.pos:
                self.planned_target = (None, target_pos)

    def get_random_target2(self, allow_visited=True):
        graph_nodes = self.model.graph.nodes() #perform set like operations, finds shortest path 

        known_pos = set(self.known_tiles.keys())
        #isinstance(known_pos, tuple)
        # print(type(known_pos))
        # print(type(self.visited_tiles))
        # print(type(self.visible_tiles))
        #print("previous known_pos", known_pos)
        #print("len prev_known_pos", len(known_pos))
        #print(self.visited_tiles)
        # If we are excluding visited tiles, remove the visited_tiles set from the available tiles
        if not allow_visited:
            #print(self.visited_tiles)
            known_pos -= self.visited_tiles

        water_coord = []
        for pos,agents in self.known_tiles.items():
            #water_coord = []
            #print(self.known_tiles.items())
            #print(agents, pos)
            for agent in agents:
                #print(agent)
                if isinstance(agent, Water):
                    x = agent.get_position()
                    #print("The water location is :",x)
                    water_coord.append(x)
        #print("List of water coord:",water_coord)

        water_coord_set = set(water_coord)
        #print("water coord as a set", water_coord_set)

        #for i in water_coord_set:
            #print(water_coord_set)
        #print("initial known pos",known_pos)
            #if i in known_pos:
                #known_pos -= water_coord_set
                #index = known_pos.index(i)
                #print("index of coord is:",index)

        new_pos = [i for i in known_pos if i not in water_coord_set]
        #print("pos after deletion",new_pos)
        #print("length of del_pos", len(new_pos))

           

        traversable_pos = [pos for pos in new_pos if self.location_is_traversable(pos)]
        # print(type(traversable_pos))
        #print("new traversable pos", traversable_pos)
        #print("length of trav pos", len(traversable_pos))
        
        while not self.planned_target[1]:
            print("move away from water")
            if len(traversable_pos) <= 50:
                print("access entire area again")
                traversable_pos = [pos for pos in known_pos if self.location_is_traversable(pos)]

            i = np.random.choice(len(traversable_pos))
            target_pos = traversable_pos[i]
            if target_pos in graph_nodes and target_pos != self.pos:
                self.planned_target = (None, target_pos)
            # traversable_pos = [pos for pos in known_pos if self.location_is_traversable(pos)]

            # while not self.planned_target[1]:
            #     print("moving through water to find shelter")
            #     i = np.random.choice(len(traversable_pos))
            #     target_pos = traversable_pos[i]
            #     if target_pos in graph_nodes and target_pos != self.pos:
            #         self.planned_target = (None, target_pos)





    def attempt_exit_plan(self):
        print(f"agent with id {self.unique_id} is attempting to exit")
        self.planned_target = (None, None)
        emergency_exits = set()

        #print(self.known_tiles)
        for pos, agents in self.known_tiles.items():
            for agent in agents:
                if isinstance(agent, EmergencyExit):
                    emergency_exits.add((agent, pos))

        if len(emergency_exits) > 0:
            if len(emergency_exits) > 1:  # If there is more than one exit known
                best_distance = None
                for exit, exit_pos in emergency_exits:
                    length = len(
                        get_line(self.pos, exit_pos)
                    )  # Let's use Bresenham's to find the 'closest' exit
                    if not best_distance or length < best_distance:
                        best_distance = length
                        self.planned_target = (exit, exit_pos)

            else:
                self.planned_target = emergency_exits.pop() #removes item at a given index

            #print(f"Agent {self.unique_id} found an emergency exit.", self.planned_target)
        else:  # If there's a flood and no emergency-exit in sight,  move randomly (for now)
            
            # Still didn't find a planned_target, so get a random unvisited target
            if not self.planned_target[1]:
                #print("getting random target to move")
                self.get_random_target2(allow_visited=False)

    def shortest_known_shelter(self):
        coord = []
        for  agent in self.model.schedule.agents:
            if isinstance(agent, EmergencyExit):
                print("found emergency exit")
                x = agent.get_position()
                print(x)
                coord.append(x)
                print(coord)

        if len(coord) ==0:
            print("no shelter found")
            pass

        elif len(coord) ==1:
            print("found one shelter only")
            self.planned_target = (None, coord[0])

        else:
            print("length of coord is greater than 1")
            dist = 100000
            curr = coord[0]
            for cell in coord:
                x = cell[0]
                y = cell[1]

                diff = abs(self.pos[0]-x) + abs(self.pos[1]-y)

                if diff < dist:
                    curr = cell
                    dist = diff 

            new_pos = curr
            print("closest shelter is :", new_pos)
            self.planned_target = (None,new_pos)

        return self.planned_target[1]

    def die(self):
        # Store the agent's position of death so we can remove them and place a DeadHuman
        pos = self.pos
        self.model.grid.remove_agent(self)
        dead_self = DeadHuman(pos, self.model)
        self.model.grid.place_agent(dead_self, pos)
        self.model.schedule.add(dead_self)
        print(f"Agent {self.unique_id} died at", pos)

    def health_mobility_rules(self):

        neighborhood = self.model.grid.get_neighborhood(self.pos,moore=True,include_center=False,radius=1)
        #print(neighborhood)
        contents = self.model.grid.get_cell_list_contents(self.pos)
        #print('contents', contents)
        for content in contents:
            if isinstance(content,Water):
                print('surrounding has water')
                self.die()           # if water catches up to agent, it dies. Prerviously health would decrease but agent would still live.

    def check_surrounding(self):
        if self.morale_boost:
            return
    
        for _,agents in self.visible_tiles:
            
            for agent in agents:
                if isinstance(agent,Water):
                    #print(f"Agent {self.unique_id} is self-warned !")
                    self.self_warned = True
                
                if not isinstance(agent,Water):
                    self.self_warned = False
                    #print(f"Agent {self.unique_id} inactive and unaware of the flood")

    def check_awareness(self):

        if self.route_information:
            self.awareness = Human.Awareness.AWARE
            #print(f"Agent {self.unique_id} knows shelter location")
        
        if not self.route_information:
            self.awareness = Human.Awareness.UNAWARE
            #print(f"Agent {self.unique_id} unaware of shelter location")
    
    def move_away_from_water(self):
       
        neighbourhood_area = self.model.grid.get_neighborhood(self.pos,moore=True,include_center=False,radius=1)
                
        for cell in neighbourhood_area:
            contents = self.model.grid.get_cell_list_contents((cell[0], cell[1]))
            for agent in contents:
                if not isinstance(agent,Water):
                    path = get_line(self.pos, (cell[0], cell[1]))
                    self.model.grid.move_agent(self, path)
                        

            
    def learn_environment(self):
        if self.knowledge < self.MAX_KNOWLEDGE:  # If there is still something to learn
            new_tiles = 0

            for pos, agents in self.visible_tiles:
                if pos not in self.known_tiles.keys():
                    new_tiles += 1
                self.known_tiles[pos] = set(agents)

            # update the knowledge Attribute accordingly
            total_tiles = self.model.grid.width * self.model.grid.height
            new_knowledge_percentage = new_tiles / total_tiles
            self.knowledge = self.knowledge + new_knowledge_percentage
            # print("Current knowledge:", self.knowledge)

    def verbal_collaboration(self, target_exit: Self, target_location: Coordinate):
        success = False
        for _, agents in self.visible_tiles:
            for agent in agents:
                if isinstance(agent, Human) and agent.get_mobility() == Human.Mobility.NORMAL:
                    if not agent.believes_alarm:
                        agent.set_believes(True)
                        print("agent informed through verbal collab and now believes the alarm ")
                        success = True

                        # Inform the agent of the target location
                    if not target_location in agent.known_tiles:
                        agent.known_tiles[target_location] = set()
                        success = True
                        print(f"agent {agent.unique_id} did not know of location previously")
                    
                    if target_exit not in agent.known_tiles[target_location]:       #setting success to true only if the agent did not already know about exit
                        agent.known_tiles[target_location].add(target_exit)         # and thus self was able to give this agent a new piece of info, which is actual collaboration
                        success = True
                        print("agent did not know of exit previously")

        if success:
            #print("Agent informed others of an emergency exit!")
            self.verbal_collaboration_count += 1

    def check_for_collaboration(self):
                
        # if self.test_collaboration():
        for location, visible_agents in self.visible_tiles:
            
            for agent in visible_agents:
                    
                if isinstance(agent, EmergencyExit):
                    # Verbal collaboration
                    self.planned_action = Human.Action.VERBAL_SUPPORT
                    self.verbal_collaboration(agent, location)


    def get_next_location(self, path):
        path_length = len(path)
        speed_int = int(np.round(self.speed))

        try:
            if path_length <= speed_int:
                next_location = path[path_length - 1]
            else:
                next_location = path[speed_int]

            next_path = []
            for location in path:
                next_path.append(location)
                if location == next_location:
                    break

            return (next_location, next_path)
        except Exception as e:
            raise Exception(
                f"Failed to get next location: {e}\nPath: {path},\nlen: {len(path)},\nSpeed: {self.speed}"
            )

    def get_path(self, graph, target, include_target=True) -> list[Coordinate]:
        path = []
        visible_tiles_pos = [pos for pos, _ in self.visible_tiles]

        try:
            if target in visible_tiles_pos:  # Target is visible, so simply take the shortest path
                path = nx.shortest_path(graph, self.pos, target)
            else:  # Target is not visible, so do less efficient pathing
                # TODO: In the future this could be replaced with a more naive path algorithm
                path = nx.shortest_path(graph, self.pos, target)

                if not include_target:
                    del path[
                        -1
                    ]  # We don't want the target included in the path, so delete the last element

            return list(path)
        except nx.exception.NodeNotFound as e:
            graph_nodes = graph.nodes()

            if target not in graph_nodes:
                contents = self.model.grid.get_cell_list_contents(target)
                print(f"Target node not found! Expected {target}, with contents {contents}")
                return path
            elif self.pos not in graph_nodes:
                contents = self.model.grid.get_cell_list_contents(self.pos)
                raise Exception(
                    f"Current position not found!\nPosition: {self.pos},\nContents: {contents}"
                )
            else:
                raise e

        except nx.exception.NetworkXNoPath as e:
            print(f"No path between nodes! ({self.pos} -> {target})")
            return path

    def location_is_traversable(self, pos) -> bool:
        if not self.model.grid.is_cell_empty(pos):
            contents = self.model.grid.get_cell_list_contents(pos)
            for agent in contents:
                if not agent.traversable:
                    return False

        return True


    def update_target(self):
        # If there was a target agent, check if target has moved or still exists
        planned_agent = self.planned_target[0]
        if planned_agent:
            current_pos = planned_agent.get_position()
            if current_pos and current_pos != self.planned_target[1]:  # Agent has moved
                self.planned_target = (planned_agent, current_pos)
                # print("Target agent moved. Updating current position:", self.planned_target)
            elif not current_pos:  # Agent no longer exists
                # print("Target agent no longer exists. Dropping.", self.planned_target, current_pos)
                self.planned_target = (None, None)
                self.planned_action = None

    def update_action(self):
        planned_agent, _ = self.planned_target

        if planned_agent:
            # Agent had planned verbal collaboration, but the agent is no longer alive, so drop it.
            if self.planned_action == Human.Action.VERBAL_SUPPORT and (
                not planned_agent.get_status() == Human.Status.ALIVE
            ):
                # print("Target agent no longer panicking. Dropping action.")
                self.planned_target = (None, None)
                self.planned_action = None
           

    def perform_action(self):
        agent, _ = self.planned_target

        if self.planned_action == Human.Action.MORALE_SUPPORT:
            # Attempt to give the agent a permanent morale boost according to your experience score
            if agent.attempt_morale_boost(self.experience):
                print("Morale boost succeeded")
            else:
                print("Morale boost failed")

            self.morale_collaboration_count += 1

        self.planned_action = None

    

    def move_toward_target(self):
        next_location: Coordinate = None
        pruned_edges = set()
        graph = deepcopy(self.model.graph)

        self.update_target()  # Get the latest location of a target, if it still exists
        if self.planned_action:  # And if there's an action, check if it's still possible
            self.update_action()

        while self.planned_target[1] and not next_location:
            if self.location_is_traversable(self.planned_target[1]):
                # Target is traversable
                path = self.get_path(graph, self.planned_target[1])
            else:
                # Target is not traversable (e.g. we are going to another Human), so don't include target in the path
                path = self.get_path(graph, self.planned_target[1], include_target=False)

            if len(path) > 0:
                next_location, next_path = self.get_next_location(path)

                if next_location == self.pos:
                    continue

                if next_location == None:
                    raise Exception("Next location can't be none")

                # Test the next location to see if we can move there
                if self.location_is_traversable(next_location):
                    # Move normally
                    self.previous_pos = self.pos
                    self.model.grid.move_agent(self, next_location)
                    self.visited_tiles.add(next_location)

                    
                elif self.pos == path[-1]:
                    # The human reached their target!

                    if self.planned_action:
                        self.perform_action()

                    self.planned_target = (None, None)
                    self.planned_action = None
                    break

                 

            else:  # No path is possible, so drop the target
                self.planned_target = (None, None)
                self.planned_action = None
                break

        if len(pruned_edges) > 0:
            # Add back the edges we removed when removing any non-traversable nodes from the global graph, because they may be traversable again next step
            graph.add_edges_from(list(pruned_edges))

    def step(self):

        if self.mobility_good == False:
            if self.model.curr_time.minute == 30:
                print(f"No action will be taken by agent {self.unique_id}")
                return

        if not self.escaped and self.pos:
            self.health_mobility_rules()

            if not self.pos:
                return

            self.visible_tiles = self.get_visible_tiles()

            self.learn_environment()

            planned_target_agent = self.planned_target[0]

            # If a flood has started and the agent believes it, attempt to plan an exit location if we haven't already and we aren't performing an action
            if (self.model.alarm and self.believes_alarm) or self.self_warned:    #this or statement is causing problems 
                #print("checking for shelter awareness")
                self.check_awareness()
                if self.mobility == Human.Mobility.NORMAL and self.awareness == Human.Awareness.AWARE :
                    
                    if not isinstance(planned_target_agent, EmergencyExit) and not self.planned_action:
                        #print(f"agent {self.unique_id} believes the alarm and attempts to escape")
                        self.attempt_exit_plan()
                       

                # Check if anything in vision can be collaborated with, if the agent has normal mobility
                    if self.mobility == Human.Mobility.NORMAL and self.collaborates:
                        print(f"agent {self.unique_id} attempts collaboration")
                        self.check_for_collaboration()

                #self.check_awareness()
                elif self.mobility == Human.Mobility.NORMAL and self.awareness == Human.Awareness.UNAWARE :
                    #print(f"Agent {self.unique_id} sets random target")
                    #num = 0.2
                    #if random.uniform(0,1) > num:
                        #print("threshold met for shelter route")
                    #print(f"Agent {self.unique_id} sets random target")
                    #self.move_away_from_water()
                    self.attempt_exit_plan()
                    #print("get random target")
                    #self.get_random_target()


            #planned_pos = self.planned_target[1]
            #self.move_toward_target()
            
            

        ##if one does not beleive the alarm 
            if not self.believes_alarm:
                #print(f"alarm not released or agent {self.unique_id} does not believe the alarm but agent checks surounding")
              # 
                self.check_surrounding()

                if self.self_warned == True :
                    #print(f"Agent {self.unique_id} now believes the flood is real!")
                    #self.check_for_collaboration()
                    self.check_awareness()

                    #if self.mobility == Human.Mobility.NORMAL and self.awareness == Human.Awareness.AWARE:
                #if self.mobility == Human.Mobility.NORMAL and self.route_information:
                        #print(f"agent {self.unique_id} attempts to escape")
                        #self.attempt_exit_plan()
                        #planned_pos = self.planned_target[1]
                        #self.move_toward_target()

                    #elif self.mobility == Human.Mobility.NORMAL and self.awareness == Human.Awareness.UNAWARE:
                        #print(f" agent {self.unique_id} mobility is normal but agent unaware of shelter - random movement away from water")
                        #num = 0.5
                        #if random.uniform(0,1) > num:
                            #self.get_random_target()
                        #self.get_random_target()
                        #self.move_away_from_water()

            planned_pos = self.planned_target[1]
            self.move_toward_target()   

            # Agent reached a shelter, proceed to exit
            if self.model.flood_started and self.pos in self.model.flood_exits.keys():
                
                self.escaped = True
                print(f"Agent {self.unique_id} has reached shelter")
                self.model.grid.remove_agent(self)

    # def get_status(self):                             # we need to redefine this get status function 
    #     if self.health > self.MIN_HEALTH and not self.escaped:
    #         return Human.Status.ALIVE
    #     elif self.health <= self.MIN_HEALTH and not self.escaped:
    #         return Human.Status.DEAD
    #     elif self.escaped:
    #         return Human.Status.ESCAPED


    #     return None

    def get_status(self):
        if not self.escaped:
            return Human.Status.ALIVE
        else:
            return Human.Status.ESCAPED

    def get_speed(self):
        return self.speed

    def get_mobility(self):
        return self.mobility

    def get_health(self):
        return self.health

    def get_position(self):
        return self.pos

    def get_plan(self):
        return (self.planned_target, self.planned_action)

    def set_plan(self, agent, location):
        self.planned_action = None
        self.planned_target = (agent, location)

    def set_health(self, value: float):
        self.health = value

    def set_believes(self, value: bool):
        if value and not self.believes_alarm:
            print(f"Agent {self.unique_id} told to believe the alarm!")

        self.believes_alarm = value

    def attempt_morale_boost(self, experience: int):
        rand = np.random.random()
        if rand < (experience / self.MAX_EXPERIENCE):
            self.morale_boost = True
            self.mobility = Human.Mobility.NORMAL
            return True
        else:
            return False
   
    def get_verbal_collaboration_count(self):
        return self.verbal_collaboration_count

    

    
    



    


    