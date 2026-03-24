# code to run the MDP agent on the dungeon environment = Value Iteration MDP Agent
# This is a simple agent that uses value iteration to compute the optimal policy for the dungeon environment
# It then follows the optimal policy to navigate the dungeon and collect treasure while avoiding monsters
# It has a full knowledge of the environment and does not need to explore

import numpy as np
import random
from itertools import product
from dungeon_env import (DungeonEnv, CELL_WALL, ACTIONS,)

GAMMA = 0.95 # discount factor for value iteration - CAN ADJUST
THETA = 1e-3 # convergence threshold for value iteration - CAN ADJUST
MAX_ITERATIONS = 200 # maximum number of iterations for value iteration - CAN ADJUST

#values to shape the reward function of the MDP (Had to add these to get things to work after a LOT of errors)
REWARD_STEP = -0.1 # small negative reward for each step taken (to encourage shorter paths)
REWARD_TREASURE = 200.0 # Expected mean treasure reward
REWARD_MONSTER_KILL = 45.0 # Expected mean monster kill reward
REWARD_EXIT_BONUS = 150.0 # Bonus for exiting the dungeon with treasure
REWARD_DEATH = -200.0 # Penalty for dying (HP reaching 0) - Death is bad

class MDPAgent:
    def __init__(self, env: DungeonEnv, gamma = GAMMA, theta = THETA):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = {} #state value
        self.pi = {} #policy
        self._solved = False #also added to more easily track whether the agent has solved the MDP yet - Look for better method?
        self._exit_pos = None
        self._last_sig = None
        self._initial_n_treasure = None

    def solve(self):
        # run value iteration on dungeon layout to compute optimal policy (use full layout)
        env = self.env
        self._exit_pos = env.exit_pos
        if self._initial_n_treasure is None:
            self._initial_n_treasure = len(env.treasure_positions)
        
        start_treasures = frozenset(env.treasure_positions)
        start_monsters = frozenset(m.pos for m in env.monsters if m.alive)
        start_state = (env.agent_pos, start_treasures, start_monsters)
        
        # print("[MDP] Starting value iteration...", end=' ', flush = True)
        reachable = set()
        queue = [start_state]
        reachable.add(start_state)
        
        while queue:
            state = queue.pop()
            for next_state, _, _ in self._transitions(state):
                if next_state not in reachable and next_state is not None:
                    reachable.add(next_state)
                    queue.append(next_state)
        
        states = list(reachable)
        
        for s in states:
            self.V[s] = 0.0 #initialize all state values to 0
        
        
        for iteration in range(MAX_ITERATIONS):
            delta = 0.0
            for s in states:
                best_val = float('-inf')
                best_action = 'up' # I defaulted to up because attack was not working for some reason
                for next_state, reward, action in self._transitions(s):
                    val = reward if next_state is None else reward + self.gamma * self.V.get(next_state, 0.0)
                    if val > best_val:
                        best_val    = val
                        best_action = action
                delta         = max(delta, abs(best_val - self.V.get(s, 0.0)))
                self.V[s]     = best_val
                self.pi[s]    = best_action
                
            if delta < self.theta:
                # print(f"[MDP] Value iteration converged after {iteration+1} iterations.")
                break
        else:
            print(f"[MDP] Value iteration reached max iterations ({MAX_ITERATIONS}) without full convergence.")
            
        self._solved = True
        
    def _transitions(self, state):
        # given a state, return a list of next_states, rewards and actions for all possible actions (use full layout)
        agent_pos, treasure_pos, monsters_pos = state
        results = []
        
        for action, (dr, dc) in ACTIONS.items():
            r, c = agent_pos
            nr, nc = r + dr, c + dc
            if not self._walkable(nr, nc) or (nr, nc) in monsters_pos:
                results.append((state, REWARD_STEP, action)) #invalid move, stay in same state with step penalty
                continue
            
            new_pos = (nr, nc)
            new_treasures = treasure_pos
            reward = REWARD_STEP
            
            if new_pos in treasure_pos:
                new_treasures = treasure_pos - {new_pos}
                reward += REWARD_TREASURE
                
            results.append(((new_pos, new_treasures, monsters_pos), reward, action))
        
        # Add attack action if adjacent to a monster
        adj_monsters = [mp for mp in monsters_pos if self._is_adjacent(agent_pos, mp)]
        if adj_monsters:
            target = adj_monsters[0] # attack the first adjacent monster
            new_monsters = monsters_pos - {target}
            results.append(((agent_pos, treasure_pos, new_monsters), REWARD_STEP + REWARD_MONSTER_KILL, 'attack'))
        else:
            results.append((state, REWARD_STEP, 'attack')) # attack with no adjacent monster does nothing but costs a step
            
        # Add exit action if on exit cell
        if agent_pos == self._exit_pos:
            treasures_collected = (self._initial_n_treasure or 0) - len(treasure_pos)
            if treasures_collected > 0:
                results.append((None, REWARD_EXIT_BONUS, 'exit')) # exiting with treasure ends episode with bonus reward
            else:
                results.append((state, REWARD_STEP, 'exit')) # exiting with no treasure just ends episode with step penalty
        else:
            results.append((state, REWARD_STEP, 'exit')) # trying to exit from non-exit cell does nothing but costs a step
            
        return results
    
    def act(self, obs, info):
        #action code to find best action given observed state and info (use full layout)
        if not self._solved:
            self.solve()
            
        agent_pos, exit_pos, treasure_pos, monster_pos = obs
        state = (agent_pos, frozenset(treasure_pos), frozenset(monster_pos))
        
        #resolve for new world
        sig = (frozenset(treasure_pos), frozenset(monster_pos))
        if sig != self._last_sig:
            self._last_sig = sig
            self.solve() #re-solve MDP if treasure or monster configuration has changed (e.g. after picking up treasure or killing monster)
            state = (agent_pos, frozenset(treasure_pos), frozenset(monster_pos))
            
        action = self.pi.get(state) or self._greedy_fallback(state)
        return action
    
    def _greedy_fallback(self, state): 
        #added this as a fallback to solve some error states (probably could be improved)
        best_val, best_act = float('-inf'), 'up'
        for next_state, reward, action in self._transitions(state):
            val = reward + self.gamma * self.V.get(next_state, 0.0)
            if val > best_val:
                best_val = val
                best_act = action
        return best_act
    
    def _walkable(self, r, c):
        if not (0 <= r < self.env.size and 0 <= c < self.env.size):
            return False
        return self.env.base_grid[r][c] != CELL_WALL
    
    def _is_adjacent(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2) == 1