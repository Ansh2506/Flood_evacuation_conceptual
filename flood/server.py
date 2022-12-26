from os import listdir, path

from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from model import FloodEvacuation
from agent import EmergencyExit, Wall, House, Water, Human, Sight, DeadHuman, Highway, Path, AgricultureField,Tree, Bridge,Tile, Forecaster


# Creates a visual portrayal of our model in the browser interface
def flood_evacuation_portrayal(agent):
    if agent is None:
        return

    portrayal = {}
    (x, y) = agent.get_position()
    portrayal["x"] = x
    portrayal["y"] = y

    if type(agent) is Human:
        portrayal["scale"] = 1
        portrayal["Layer"] = 5
        portrayal["Shape"] = "resources/human.png"
    elif type(agent) is Water:
        portrayal["Shape"] = "resources/water.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 3
    elif type(agent) is EmergencyExit:
        portrayal["Shape"] = "resources/flood_exit.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Wall:
        portrayal["Shape"] = "resources/wall.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is House:
        portrayal["Shape"] = "resources/building.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Tree:
        portrayal["Shape"] = "resources/tree.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Bridge:
        portrayal["Shape"] = "resources/bridge.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Highway:
        portrayal["Shape"] = "resources/highway.jpg"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Path:
        portrayal["Shape"] = "resources/PATH.jpg"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is AgricultureField:
        portrayal["Shape"] = "resources/field.jpg"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is DeadHuman:
        portrayal["Shape"] = "resources/dead.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 4
    elif type(agent) is Sight:
        portrayal["Shape"] = "resources/eye.png"
        portrayal["scale"] = 0.8
        portrayal["Layer"] = 7
    elif type(agent) is Tile:
        portrayal["Layer"] = 1
        portrayal["scale"] = 1

    elif type(agent) is Forecaster:
        portrayal["Layer"] = 1
        portrayal["scale"] = 1
        
    
    
    return portrayal


# Was hoping floorplan could dictate the size of the grid, but seems the grid needs to be specified first, so the size is fixed to 50x50
canvas_element = CanvasGrid(flood_evacuation_portrayal, 50, 50, 800, 800)

# Define the charts on our web interface visualisation
status_chart = ChartModule(
    [
        {"Label": "Alive", "Color": "blue"},
        {"Label": "Dead", "Color": "red"},
        {"Label": "Escaped", "Color": "green"},
    ]
)

collaboration_chart = ChartModule(
    [
        {"Label": "Verbal Collaboration", "Color": "orange"},
        
    ]
)

# Get list of available floorplans
floor_plans = [
    f
    for f in listdir("floorplans")
    if path.isfile(path.join("floorplans", f))
]

# Specify the parameters changeable by the user, in the web interface
model_params = {
    "floor_plan_file": UserSettableParameter(
        "choice", "Floorplan", value=floor_plans[4], choices=floor_plans
    ),
    "human_count": UserSettableParameter("number", "Number Of Human Agents", value=10
    ),
    "collaboration_percentage": UserSettableParameter(
        "slider", "Percentage Collaborating", value=60, min_value=0, max_value=100, step=10
    ),
    "flood_probability": UserSettableParameter(
        "slider", "Probability of Flood", value=0.8, min_value=0, max_value=1, step=0.01
    ),
    "random_spawn": UserSettableParameter(
        "checkbox", "Spawn Agents at Random Locations", value=False
    ),
    "visualise_vision": UserSettableParameter("checkbox", "Show Agent Vision", value=False),
    "save_plots": UserSettableParameter("checkbox", "Save plots to file", value=True
    ),
    "route_information_percentage": UserSettableParameter(
        "slider", "Percentage Aware of Route", value=20, min_value=0, max_value=100, step=10
    ),
     "mobility_good_percentage": UserSettableParameter(
        "slider", "Percentage with good mobility", value=80, min_value=0, max_value=100, step=10
    )
    }

# Start the visual server with the model
server = ModularServer(
    FloodEvacuation,
    [canvas_element, status_chart, collaboration_chart],
    "Flood Evacuation",
    model_params,
)
