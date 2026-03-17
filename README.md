# Dungeon Crawler
### Algorithms for Machine Design Project
#### Austin J. Erickson

## Description
A repository for my Algorithms for Decision Making class project. The project involves coding up a Dungeon crawler and having a decision-making algorithm solve it. Can also be played by a human.

## Rules

1. Agent's goal is to obtain as much treasure as possible and exit the dungeon
2. Treasure rewards will range from 50-500 positive points
2. Monsters will seek to stop the agent
3. If the agent reaches 0 hp he dies and banks nothing
4. Defeating monsters gives 100 positive points
5. Monsters remain motionless unless they see the agent. Upon seeing an agent they will move towards the agent, so long as there is an unbroken line of sight.
6. If the monster loses sight of the agent for two "ticks" (movements) it stops moving
5. Agent can only see X tiles away (X being defined in run_game)
8. Agent ONLY gets positive points for reaching the exit with treasure. No penalty for lost hp.
9. Failure to exit the dungeon without treasure results in a score of 0, making the scores range from 0 - (Total treasure and killed monsters)
