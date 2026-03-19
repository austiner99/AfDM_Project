# Dungeon Crawler
### Algorithms for Machine Design Project
#### Austin J. Erickson

## Description
A repository for my Algorithms for Decision Making class project. The project involves coding up a Dungeon crawler and having a decision-making algorithm solve it. Can also be played by a human.

The following algorithms were created to play this game:

### MDP Value Function
A simple value function who is able to "cheat" or gain a full knowledge of the floor setup, monster positions,  treasure positions and the exit. It will precompute an "optimal" path but be unable to adjust based on state changes.

### MCTS
A Monte Carlo Tree Search Algorithm that will also have a full knowledge of the floor, but will be able to adjust to state changes and make decisions at every step.

### POMCP
A Partially Observed MCTS algorithm which will use a particle filter to estimate state and actions based on limited observed information.

## Rules

1. Agent's goal is to obtain as much treasure as possible and exit the dungeon
2. Agent can move in any cardinal direction, attack or exit (if applicable)
2. Treasure rewards will range from 50-500 positive points
2. Monsters will seek to stop the agent
3. Monsters always attack first
3. If the agent reaches 0 hp he dies and banks nothing
4. Defeating monsters gives points equal to how difficult they were to defeat
5. Monsters remain motionless unless they see the agent. Upon seeing an agent they will move towards the agent, so long as there is an unbroken line of sight.
6. If the monster loses sight of the agent for three "ticks" (movements) it becomes idle again
5. Agent can only see three tiles away
8. Agent ONLY gets positive points for reaching the exit with treasure. No penalty for lost hp.
9. Failure to exit the dungeon without treasure results in a score of 0, making the scores range from 0 - (Total treasure and killed monsters)

## Configuration

- Adjustable parameters (e.g., treasure values, visibility range, HP) can be modified in `dungeon_env.py`
- The number of monsters and treasures can be configured in `play.py`

## How to Run Game and Algorithms
### Human-Played Game
Run the `play.py` file. A window will appear displaying the game. Enter commands into the terminal to play.

### MDP-Played Game

### MCTS-Played Game

### POMCP-Played Game

## AI Disclosure

Github copilot autocompete was used to assist in autocompletion of lines and code (under heavy scrutiny). Furthermore, claude was used to clean up (remove unneccessary lines) and debug code, as well as add small annotations to various sections for easier understanding. 

In summary all major functions, code structure and algorithms were human made, and only AI-edited as needed.