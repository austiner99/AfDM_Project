#Monte carlo tree search agent for the dungeon environment
import math
import random
import copy
from dungeon_env import DungeonEnv, ACTIONS

#Tunable parameters for MCTS
NUM_SIMS = 200 # simulation per act()
MAX_DEPTH = 120 #maximum rollout depth - needs to be long enough to "see" exit (at least 60ish)
UCB_C = 1.41 #exploration constant for UCB formula - CAN ADJUST (is sqrt(2) by default)
GAMMA = 0.97 #discount factor for future rewards in MCTS - CAN ADJUST - had to make higher to actually want to get to the exit

#rollout eval:
REWARD_STEP = -0.1 # small negative reward for each step taken (to encourage shorter paths)
REWARD_TREASURE_STEP = 2.0 # reward per step closer to nearest treasure
REWARD_EXIT_STEP = 3.0 # reward per step closer to exit (if have treasure)
REWARD_EXIT_BONUS = 50.0 # Bonus for exiting the dungeon with treasure

class MCTSNode:
    #one node in the MCTS tree
    
    __slots__ = ['action', 'parent', 'children', 'visits', 'value', 'untried_actions']
    
    def __init__(self, action=None, parent=None, untried_actions=None):
        self.action = action
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = list(untried_actions or [])

        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def ucb1(self, c=UCB_C):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits) #Main MCTS formula
    
    def best_child(self, c=UCB_C):
        return max(self.children.values(), key=lambda n: n.ucb1(c)) # select child with highest UCB score
    
    def most_visited_child(self):
        return max(self.children.values(), key=lambda n: n.visits) # select child with most visits
    
class MCTSAgent:
    def __init__(self, env:DungeonEnv, num_sims = NUM_SIMS, max_depth = MAX_DEPTH, c = UCB_C, gamma = GAMMA):
        self.env = env
        self.num_sims = num_sims
        self.max_depth = max_depth
        self.c = c
        self.gamma = gamma
        self._recent_positions = []
        self._history_len = 7 #number of recent positions to track for loop detection
        
    def act(self, obs=None, info=None):
        self._recent_positions.append(self.env.agent_pos)
        if len(self._recent_positions) > self._history_len:
            self._recent_positions.pop(0)
        root_actions = self._valid_actions(self.env)
        root = MCTSNode(untried_actions=root_actions)
        
        for _ in range(self.num_sims):
            sim_env = copy.deepcopy(self.env)
            node = root
            depth = 0
            discount = 1.0
            total_reward = 0.0
            
            #selection
            while (node.is_fully_expanded() and node.children and not sim_env.done):
                node = node.best_child(self.c)
                _, reward, _, _ = sim_env.step(node.action)
                total_reward += discount * reward
                discount *= self.gamma
                depth += 1
                
                #expansion
            if not sim_env.done and node.untried_actions:
                action = node.untried_actions.pop()
                _, reward, _, _ = sim_env.step(action)
                total_reward += discount * reward
                discount *= self.gamma
                depth += 1
                
                child = MCTSNode(action = action, parent = node, untried_actions = self._valid_actions(sim_env))
                node.children[action] = child
                node = child
                
            #rollout
            total_reward += self._rollout(sim_env, depth, discount)
            
            #backpropagation
            while node is not None:
                node.visits += 1
                node.value += total_reward
                node = node.parent
        
        if not root.children:
            return random.choice(root_actions or ['up']) #if we never expanded any children, just pick a random valid action or up if there is an issue
        return root.most_visited_child().action #return the action of the most visited child of the root
    
    def _rollout(self, sim_env, current_depth, discount):
        #current rollout policy
        total_reward = 0.0
        
        for _ in range(self.max_depth - current_depth):
            if sim_env.done:
                break
            action = self._rollout_policy(sim_env)
            _, reward, _, _ = sim_env.step(action)
            total_reward += discount*reward
            discount *= self.gamma
        
        total_reward += discount *self._evaluate(sim_env) #final evaluation of the state we end up in after the rollout
        return total_reward
    
    def _rollout_policy(self, sim_env):
        pos = sim_env.agent_pos
        
        #following is done to make the code run quicker and evaluate better
        if pos == sim_env.exit_pos and sim_env.treasure_held > 0:
            return 'exit'
        if sim_env._adjacent_monsters(pos):
            return 'attack'
        
        best_action = None
        best_value = float('-inf')
        monster_cells = {m.pos for m in sim_env.monsters if m.alive}
        
        for action, (dr, dc) in ACTIONS.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if not sim_env._is_walkable(nr, nc) or (nr, nc) in monster_cells:
                continue
            value = self._step_potential(sim_env, pos, (nr, nc))
            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action if best_action is not None else random.choice(self._valid_actions(sim_env) or ['up']) #if no valid actions, just pick up to avoid crashing
    
    def _step_potential(self, sim_env, old_pos, new_pos):
        #mirror MDP for better comparison
        reward = REWARD_STEP
        
        if new_pos in self._recent_positions:
            reward -= 1.0*self._recent_positions.count(new_pos) #discourage loops by penalizing recently visited positions, more if visited multiple times in recent history
        
        treasures = sim_env.treasure_positions
        
        if treasures:
            old_dist = min(self._manhattan(old_pos, t) for t in treasures)
            new_dist = min(self._manhattan(new_pos, t) for t in treasures)
            reward += REWARD_TREASURE_STEP * (old_dist - new_dist)
            
        old_exit_dist = self._manhattan(old_pos, sim_env.exit_pos)
        new_exit_dist = self._manhattan(new_pos, sim_env.exit_pos)
        
        #this is clever and I'm proud of it - scale exit pull by how much treasure we have
        treasure_factor = 1.0 + sim_env.treasure_held/200
        reward += REWARD_EXIT_STEP * treasure_factor * (old_exit_dist - new_exit_dist)
        
        if new_pos == sim_env.exit_pos and sim_env.treasure_held > 0:
            reward += REWARD_EXIT_BONUS
            
        return reward
    
    def _evaluate(self, sim_env):
        if sim_env.done:
            return 0.0
        
        pos = sim_env.agent_pos
        treasures = sim_env.treasure_positions
        
        score = sim_env.treasure_held * 0.5
        
        if treasures:
            nearest_t = min(self._manhattan(pos, t) for t in treasures)
            score += REWARD_TREASURE_STEP * (20-nearest_t)
        else:
            exit_dist = self._manhattan(pos, sim_env.exit_pos)
            score += REWARD_EXIT_STEP * (20-exit_dist)*(1.0+sim_env.treasure_held/200)
            
        return score
    
    def _valid_actions(self, env):
        actions = []
        monster_cells = {m.pos for m in env.monsters if m.alive}
        pos = env.agent_pos
        for action, (dr, dc) in ACTIONS.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if env._is_walkable(nr, nc) and (nr, nc) not in monster_cells:
                actions.append(action)
        if env._adjacent_monsters(pos):
            actions.append('attack')
        if pos == env.exit_pos and env.treasure_held > 0:
            actions.append('exit')
        return actions if actions else list(ACTIONS.keys()) #if no valid actions, return all actions to avoid crashing
    
    @staticmethod
    def _manhattan(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])