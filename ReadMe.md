
################### INSTRUCTIONS ####################

0-Extract all zip files into the same folder
1-import condaENV.yaml into an anaconda environment and launch spyder 
2-open the DQNALE.py script
3-choose what game to play by changing the currentgame parameter 
4-set load to True to load a trained model
5-set visual to True to view the agent playing the game
6-Set debug to True to view the segmentation results
7-run the script


################### SCRIPT DESCRIOTIONS ####################

DQNALE.py: the main script that prepares the game environment and operates the DQN and the segmentation code

ai.py: the architecture of the DQN code, manages its parameters and its learning, also used to retrieve the next action

segmenttest.py: performs morphological operations to segment the game environment and prepares the data for the DQN, takes frame image as input and returns an array of object descriptors (X,Y,Class)

classify.py: stores the known detected classes of the game, performs euclidean distance on detected objects to classify them, takes an object as input and returns the class

 
################### PARAMETERS ####################

frameskips: the number of frames using the same action before asking the DQN to provide a new action
datatype: positive integer

EP: the number of episodes/iterations used for training the model
datatype: positive integer

EvalEP: the number of episodes/iterations used for evaluation
datatype: positive integer

movements: maximum number of timesteps allowed for each episode before forcefully ending the game
datatype: positive integer

targetupdate: the number of episodes to wait before updating the target network parameters
datatype: positive integer

save: toggle whether to save the DQN model parameters at the end of training or not
datatype: boolean

load: toggle whether to load existing DQN model parameters before playing
datatype: boolean

visual: toggle whether to show the game window
datatype: boolean

games: a list of ids for the selected atari games
datatype: list of strings (unchangeable)

current game: the game selection chosen
datatype: and an integer between 0 and 4 inclusively 

#SEGMENTATION

debug: toggle on to view the images resulting by each stage of the segmentation code
datatype: boolean

saveimgs: toggle on to save the images of the segmented objects into the objctimg folder
datatype: boolean

objctsize: the resolution to resize the images of detected objects
datatype: positive integer 

gridsize: size to rescale the game environment image into
datatype: positive integer

#classification

classthresh: a threshold that determines sensitivity that the classification algorithm will use to add objects to a class
datatype: positive float

#DQN
LR: learning rate of the DQN
datatype: positive float

batchsize: number of game states used for learning in one timestep in the dqn (replay memory)
datatype: positive integer


################### KNOWN BUGS ####################

if any of the bugs below occur, restart the kernel and run the program again

-If the error values initialise at rates higher than 10^3

