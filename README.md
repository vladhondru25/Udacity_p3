# Udacity_p3

1. Environment details

The environment consists of a 2D tennis game, where two rackets try to hit the ball over the net. If an agent succesfully hits the ball and it bounces over the net, it receives a rewards of postiive +0.1, otherwise, a negative 0.1 is received. The aim of this environment is to have both rackets continuosuly keep the ball in play. The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. The action space is represented by 2 variables, one for moving away or close to the net and the other for jumping. The environment is considered solved when the average score over 100 episodes is at greater than postiive 0.5.

2. Requirements

The first step is to activate the environment. Firstly, follow the instructions in the DRLND Gtihub repository to set up the Python environment [click here](https://github.com/udacity/deep-reinforcement-learning#dependencies). Following the instructions in the README.md file will install the necessary depenedencies (e.g. PyTorch, ML-Agents toolkit). 

The second step is to install the Unity Environment in which the actual game takes place. Depending on your OS, you can download it from the below links, and then place it in the main folder:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


3. How to run the code?

The main file is represented by the jupyer notebook "P3_CaC.ipynb", while other python files are used for splitting the functionalities. Each code cell has an intruction cell, which describes its functionality. The code can be run by simply running each cell in the respective order they appear.
