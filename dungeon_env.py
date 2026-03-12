#class that generates a random dungeon environment for the agent to explore
import numpy as np
import random

class DungeonEnv:
    def __init__(self, size=10, num_traps=3, num_treasures=2, num_monsters=2):
        self.size = size
        self.num_traps = num_traps
        self.num_treasures = num_treasures
        self.num_monsters = num_monsters
        self.grid = np.zeros((size, size), dtype=int)
        self.agent_pos = (0, 0)  # Starting position of the agent
        self.exit_pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))  # Goal position
