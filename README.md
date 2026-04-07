# Dungeon Crawler
### Algorithms for Machine Design Project
#### Austin J. Erickson

## Description
A repository for my Algorithms for Decision Making class project. The project involves coding up a Dungeon crawler and having a decision-making algorithm solve it. Can also be played by a human.

The following algorithms were created to play this game:

### MDP Value Function
A simple value function who is able to "cheat" or gain a full knowledge of the floor setup, monster positions,  treasure positions and the exit. It will precompute an "optimal" path but be unable to adjust based on state changes.

The other major difference between this agent and the other two is that, due to hp being a discrete value from 0-100, that portion of the state is not added here in order to save on calculation time (it exploads when taking hp into account). Because of this, the agent will always attack monsters, since it doesn't have a "fear" of losing all its health.

### MCTS
A Monte Carlo Tree Search Algorithm that will also have a full knowledge of the floor, but will be able to adjust to state changes and make decisions at every step.

### POMCP
A Partially Observed MCTS algorithm which will use a particle filter to estimate state and actions based on limited observed information. Other than that it will follow identical logic to the MCTS.

## Rules

1. Agent's goal is to obtain as much treasure as possible and exit the dungeon
2. Agent can move in any cardinal direction, attack or exit (if applicable)
2. Treasure rewards will range from 50-500 positive points
2. Monsters will seek to stop the agent
3. Monsters always attack first
3. If the agent reaches 0 hp he dies and banks nothing
4. Defeating monsters gives points equal to how difficult they were to defeat
5. Monsters remain motionless unless they see the agent. Upon seeing an agent they will move towards the agent, so long as there is an unbroken line of sight.
6. If the monster loses sight of the agent for three "steps" (movements) it becomes idle again
5. Agent can only see three tiles away, or tiles already revealed
8. Agent ONLY gets positive points for reaching the exit with treasure. No penalty for lost hp.
9. Failure to exit the dungeon without treasure results in a score of 0, making the scores range from 0 - (Total treasure and killed monsters)

## Configuration

- Adjustable parameters (e.g., treasure values, visibility range, HP, number of monsters, etc.) can be modified in `dungeon_env.py`
- The size of the playing field can be configured in `play.py` or `run_agent.py.` Unless changed, number of rooms, treasures and monsters are scaled loosely based on room size

`run_agent` parser arguments available:
| Argument | Default | Recommended | Description |
|---|---|---|---|
| `--agent` | `mdp` | `mcts` (general), `pomcp` (partial observability), `mdp` (baseline) | Agent to run (`mdp`, `mcts`, or `pomcp`) |
| `--size` | `20` | `12`–`30` | Width/height of the dungeon grid (larger sizes difficult for mdp agent) |
| `--episodes` | `1` | `1`–`50` | Number of episodes to simulate |
| `--seed` | `None` | `None` | Seed value for reproducible runs |
| `--delay` | `0.15` | `0.05`–`1` | Delay in seconds per step |
| `--no-display` | `False` | `False` (batch runs), `True` (debug/demo) | Render the game window during runs |
| `--cheat` | `False` | Either `True` or `False` | Show the full map on each render |
| `--fog` | `False` | Either `True` or `False` | Show fog (limited vision of agent/human player) |
| `--verbose` | `False` | `True` for experiments | Display updates during agent play |

## How to Run Game and Algorithms
### Human-Played Game
Run the `play.py` file. A window will appear displaying the game. Enter commands into the terminal followed by `Enter` to play. (Future developments will allow for easier play, but this was added later).

### Agnent-Played Game
Run the following command in the terminal when in the project directory:

`python run_agent.py` OR `python3 run_agent.py` followed by whatever arguments you want to add (see [Configuration](#configuration) above). Specifically, the `--agent mdp`, `--agent mcts`, and `--agent pomcp` will allow you to select which agent you would like to run.

Example terminal entry (when in project directory):

`python3 run_agent.py --mcts --size 20 --episodes 10 --delay 0.1 --cheat --verbose`

This would simulate a mcts agent playing 10 episodes on 10 different randomly generated dungeons of size 20 with a 0.1 second delay between each move (plus computation time). You would be able to see the full map during play and receive a detailed log of events during each run.

## AI Disclosure

Github copilot autocompete was used to assist in autocompletion of lines and code (under heavy scrutiny). Furthermore, claude was used to clean up (remove unneccessary lines) and debug code, as well as add small annotations to various sections for easier understanding. 

In summary all major functions, code structure and algorithms were human made, and only AI-edited as needed.

Certain functions helper functions were later added to fix bugs and run main agents better, and are disclosed in comments under the functions.

All AI-assisted code is annotated stating how much AI was used in its creation. 

## Summary of Project-Important Files in this Repository

### `dungeon_env.py`

Class that creates and manages dungeon objects and logic. 

### `mcts_agent.py`

Monte Carlo Tree Search agent class. Retains all logic and decision making processes for this agent.

### `mdp_agent.py`

Markov Decision Process agent class. Retains all logic and decision making processes for this agent.

### `play.py`

File available for human play.

### `pomcp_agent.py`

Partially Observable Monte Carlo Planning agent class. Retains all logic and decision making processes for this agent.

### `run_agent.py`

Code to run one or multiple iterations of an agent.

### `visualize.py`

Code to visualize the game board. Future iterations will have images in place of squares for more dynamic viewing.

## Summary of Non-Project-Important Files in this Repository (can be ignored)

### `experiment_results.json`

File of multiple runs done for comparison. Data will be analyzed in figure form in final video.

### `run_multiple_agents.py`

File used to allow multiple agents to run over the weekend to produce results in `experiment_results.json.` 50 episodes at dungeon sizes 12, 18, 24, and 30 were run. 

### `analyze_json_data.py`

Code used to generate box-and-whiskers plots for multi-agent analysis. 

### `analyze_win_rate.py`

Code used to generate win rate plot for multi-agent analysis.

## Summary of Results Seen

(For visual information, see plots generated in the "outputs" folder)

