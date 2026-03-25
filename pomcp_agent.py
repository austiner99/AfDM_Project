#code to run pomcp agent

import math
import random
import copy
from collections import deque
from dungeon_env import DungeonEnv, ACTIONS, CELL_WALL, CELL_EMPTY

#pomcp params
NUM_SIMS = 200 #number of sims act can run - same as MCTS for now
MAX_DEPTH = 120
UCB_C = 1.41
GAMMA = 0.97
NUM_PARTICLES = 125 # number of particles to maintain in the belief state - can adjust - had to increase to get decent performance, but this is the main bottleneck for speed so be careful increasing too much
MIN_PARTICLES = 25 # minimum number of particles to maintain - if belief falls below this, we will resample from the prior (uniform over all states) to add more particles - can adjust 

#rweards (same as mcts)
REWARD_STEP = -0.1
REWARD_TREASURE_STEP = 4.0
REWARD_EXIT_STEP = 4.0
REWARD_EXIT_BONUS = 50.0
REWARD_EXPLORE = 1.75 #added to encourage exploring new areas - can adjust (had to add to stop scared agents)

class Particle:
    # a hypothetical world state
    def __init__(self, monsters, hidden_treasure_positions):
        self.monsters = list(monsters)
        self.hidden_treasure_positions = list(hidden_treasure_positions)

    @property
    def monster_positions(self):
        return [pos for pos, _ in self.monsters]

    def copy(self):
        return Particle(list(self.monsters), list(self.hidden_treasure_positions))
    
class POMCPNode:

    __slots__ = ['action', 'parent', 'children', 'visits', 'value', 'untried_actions']

    def __init__(self, action=None, parent=None, untried_actions=None):
        self.action = action
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = list(untried_actions or [])

    def is_fully_expanded(self):
        return(len(self.untried_actions) == 0)
    
    def ucb1(self, c=UCB_C):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def best_child(self, c=UCB_C):
        return max(self.children.values(), key=lambda n: n.ucb1(c))
    
    def most_visited_child(self):
        return max(self.children.values(), key=lambda n: n.visits)
    
class POMCPAgent:
    def __init__(self, env:DungeonEnv, num_sims = NUM_SIMS, max_depth = MAX_DEPTH, c = UCB_C, gamma = GAMMA, num_particles = NUM_PARTICLES):
        self.env = env
        self.num_sims = num_sims
        self.max_depth = max_depth
        self.c = c
        self.gamma = gamma
        self.num_particles = num_particles
        self.particles = []
        self._dist_cache = {}
        self._last_pos = None
        self._recent_positions = []
        self._history_len = 8 #number of recent positions to track for loop detection
        self._initialized = False

    def _init_particles(self, obs):
        agent_pos, exit_pos, known_treasures, known_monsters = obs
        observed = self.env.explored
        walkable = self._get_unexplored_walkable(observed)
        self.particles = []
        self._initialized = True

        for _ in range(self.num_particles):

            #place monsters - we know the positions of any monsters we've seen, but the rest could be anywhere in the walkable area 
            monster_alert = {m.pos: m.alert for m in self.env.monsters if m.alive} 
            monster_list= [(m, monster_alert.get(m, True)) for m in known_monsters] 
            
            n_unknown_monsters = max(0, len(self.env.monsters) - len(known_monsters)) #we let the agent cheat here a little, since it knows HOW MANY monsters there are in total
            pool = [c for c in walkable if c not in[pos for pos, _ in monster_list] and c != agent_pos and c != exit_pos] 
            random.shuffle(pool)
            for i in range(min(n_unknown_monsters, len(pool))):
                monster_list.append((pool[i], False))

            # treasures - same as monsters
            n_unknown_treasures = max(0, len(self.env.treasure_positions) - len(known_treasures))
            treasure_pos = list(known_treasures)
            pool2 = [c for c in walkable if c not in treasure_pos and c not in[pos for pos, _ in monster_list]]
            random.shuffle(pool2)
            for i in range(min(n_unknown_treasures, len(pool2))):
                treasure_pos.append(pool2[i])
            
            #create the particles
            self.particles.append(Particle(monster_list, treasure_pos))

    def _update_particles(self, obs, action, reward):
        agent_pos, exit_pos, known_treasures, known_monsters = obs
        observed = self.env.explored
        known_monster_set = set(known_monsters)
        known_treasure_set = set(known_treasures)

        new_particles = []
        for particle in self.particles:
            p = particle.copy()

            new_monsters = []
            for m, alert in p.monsters:
                has_los = self._has_line_of_sight(m, agent_pos)
                if has_los and not alert:
                    alert = random.random() < self.env.monster_notice_chance
                elif not has_los and alert:
                    alert = random.random() < (1.0/self.env.monster_lost_sight_limit if hasattr(self.env, 'monster_lost_sight_limit') else 0.5)
                new_monsters.append((self._step_monster_toward(m, agent_pos) if alert else m, alert))
            p.monsters = new_monsters

            if action in ACTIONS:
                p.hidden_treasure_positions = [t for t in p.hidden_treasure_positions if t != agent_pos] #if we moved onto a treasure, it's no longer hidden

            #consistancy check for debugging - check that visable cells match obs
            consistent = True
            for mp in p.monster_positions:
                r, c = mp
                if observed[r][c]:
                    if mp not in known_monster_set:
                        consistent = False
                        break
            if not consistent:
                continue
            for treasure_pos in p.hidden_treasure_positions:
                r, c = treasure_pos
                if observed[r][c]:
                    if treasure_pos not in known_treasure_set:
                        consistent = False
                        break
            if consistent:
                new_particles.append(p)

        #resample if filter bad
        if len(new_particles) < MIN_PARTICLES:
            self._init_particles(obs)
        else:
            self.particles = random.choices(new_particles, k=self.num_particles)

    def _step_monster_toward(self, monster_pos, agent_pos):
        #mimic logic in dungeon_env - monsters with line of sight move to agent
        if not self._has_line_of_sight(monster_pos, agent_pos):
            return monster_pos
        mr, mc = monster_pos
        ar, ac = agent_pos
        dr = 0 if mr == ar else (1 if ar > mr else -1)
        dc = 0 if mc == ac else (1 if ac > mc else -1)
        candidates = []
        if dr!= 0:
            candidates.append((mr+dr, mc))
        if dc != 0:
            candidates.append((mr, mc+dc))
        candidates.append((mr,mc))
        for nr, nc in candidates:
            if self._walkable(nr, nc):
                return (nr, nc)
        return monster_pos

    def _has_line_of_sight(self, monster_pos, agent_pos):
        mr, mc = monster_pos
        ar, ac = agent_pos
        if mr!=ar and mc!=ac:
            return False
        if mr == ar:
            if abs(ac - mc) > self.env.monster_vision_radius:
                return False
            step = 1 if ac > mc else -1
            for c in range(mc+step, ac, step):
                if self.env.base_grid[mr][c] == CELL_WALL:
                    return False
            return True
        if abs(ar - mr) > self.env.monster_vision_radius:
            return False
        step = 1 if ar > mr else -1
        for r in range(mr+step, ar, step):
            if self.env.base_grid[r][mc] == CELL_WALL:
                return False
        return True
    
    def act(self, obs, info):
        #similar logic to mcts, but with belief state 
        self._last_pos = self.env.agent_pos
        self._recent_positions.append(self.env.agent_pos)
        if len(self._recent_positions) > self._history_len:
            self._recent_positions.pop(0)
        
        if not self._initialized:
            self._init_particles(obs)
        else:
            self._update_particles(obs, getattr(self, '_last_action', None), info.get('reward', 0))

        root_actions = self._valid_actions_from_obs(obs)
        root = POMCPNode(untried_actions=root_actions)

        for _ in range(self.num_sims):
            #sample a single env from this for this run
            if not self.particles:
                self._init_particles(obs)
            particle = random.choice(self.particles).copy()

            sim_env = self._particle_to_env(particle, obs)
            node = root
            depth = 0
            discount = 1.0
            total_reward = 0.0

            #selection
            while(node.is_fully_expanded() and node.children and not sim_env.done):
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

                child = POMCPNode(action=action, parent=node, untried_actions=self._valid_actions_from_env(sim_env))
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
        
        best_action = root.most_visited_child().action #return the action of the most visited child of the root
        self._last_action = best_action

        return best_action
    
    def _particle_to_env(self, particle, obs):
        agent_pos, exit_pos, known_treasures, known_monsters = obs
        sim = copy.deepcopy(self.env)
        from dungeon_env import Monster
        sim.monsters = []
        for mp, alert in particle.monsters:
            m = Monster(mp)
            m.alert = alert 
            sim.monsters.append(m)
        
        sim.treasure_positions = list(known_treasures) + [t for t in particle.hidden_treasure_positions if t not in known_treasures]

        sim._redraw_grid()
        return sim
    
    def _rollout(self, sim_env, current_depth, discount):
        #similar to mcts
        total_reward = 0.0
        visited = {sim_env.agent_pos}
        for _ in range(self.max_depth - current_depth):
            if sim_env.done:
                break
            action = self._rollout_policy(sim_env, visited)
            _, reward, _, _ = sim_env.step(action)
            total_reward += discount*reward
            visited.add(sim_env.agent_pos)
            discount *= self.gamma
        total_reward += discount*self._evaluate(sim_env)
        return total_reward
    
    def _rollout_policy(self, sim_env, visited=None):
        pos = sim_env.agent_pos
        if pos == sim_env.exit_pos and sim_env.treasure_held > 0:
            return 'exit'
        if sim_env._adjacent_monsters(pos):
            return 'attack'
        best_action = None
        best_value = float('-inf')
        monster_cells = {m.pos for m in sim_env.monsters if m.alive}
        for action, (dr, dc) in ACTIONS.items():
            nr, nc = pos[0]+dr, pos[1]+dc
            if not sim_env._is_walkable(nr, nc) or (nr, nc) in monster_cells:
                continue
            value = self._step_potential(sim_env, pos, (nr, nc))

            if visited and (nr, nc) in visited:
                value -= 2.0
            if value > best_value:
                best_value = value
                best_action = action
        return best_action if best_action else random.choice(self._valid_actions_from_env(sim_env) or ['up']) 

    def _step_potential(self, sim_env, old_pos, new_pos):
        reward = REWARD_STEP

        if new_pos ==self._last_pos:
            reward -= 5.0
        elif new_pos in self._recent_positions:
            recency = [i for i, p in enumerate(self._recent_positions) if p == new_pos]
            most_recent_idx = max(recency)
            reward -= 0.5 * (most_recent_idx+1)

        treasures = sim_env.treasure_positions
        if treasures:
            old_dist = min(self._bfs_dist(old_pos, t) for t in treasures)
            new_dist = min(self._bfs_dist(new_pos, t) for t in treasures)
            reward += REWARD_TREASURE_STEP * (old_dist - new_dist)
        old_exit_dist = self._bfs_dist(old_pos, sim_env.exit_pos)
        new_exit_dist = self._bfs_dist(new_pos, sim_env.exit_pos)
        treasures_found = sim_env.treasure_held / 200
        reward += REWARD_EXIT_STEP * (old_exit_dist - new_exit_dist)*treasures_found

        if new_pos == sim_env.exit_pos and sim_env.treasure_held > 0:
            reward += REWARD_EXIT_BONUS

        explored = getattr(sim_env, 'explored', self.env.explored)
        nearest_unexplored = self._nearest_unexplored_in(old_pos, explored)
        if nearest_unexplored is not None:
            old_unexplored_dist = self._bfs_dist(old_pos, nearest_unexplored)
            new_unexplored_dist = self._bfs_dist(new_pos, nearest_unexplored)
            explore_weight = max(0.0, 1.0 - sim_env.treasure_held/300.0) 
            reward += REWARD_EXPLORE * explore_weight * (old_unexplored_dist - new_unexplored_dist)

        return reward
    
    def _evaluate(self, sim_env):
        if sim_env.done:
            return 0.0
        pos = sim_env.agent_pos
        treasures = sim_env.treasure_positions
        score = sim_env.treasure_held * 0.5
        if treasures:
            nearest_t = min(self._bfs_dist(pos, t) for t in treasures)
            score += REWARD_TREASURE_STEP * (20-nearest_t)
        else:
            exit_dist = self._bfs_dist(pos, sim_env.exit_pos)
            score += REWARD_EXIT_STEP * (20-exit_dist)*(sim_env.treasure_held/200)

        return score
    
    def _valid_actions_from_obs(self, obs):
        agent_pos, exit_pos, known_treasures, known_monsters = obs
        actions = []
        monster_cells = set(known_monsters)
        r, c = agent_pos
        for action, (dr, dc) in ACTIONS.items():
            nr, nc = r+dr, c+dc
            if self._walkable(nr, nc) and (nr, nc) not in monster_cells:
                actions.append(action)
        for m in known_monsters:
            if abs(m[0] - r) + abs(m[1] - c) == 1:
                actions.append('attack')
                break
        if agent_pos == exit_pos and self.env.treasure_held > 0:
             actions.append('exit')
        return actions if actions else list(ACTIONS.keys()) 
    
    def _valid_actions_from_env(self, env):
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
        return actions if actions else list(ACTIONS.keys()) 
    
    def _get_unexplored_walkable(self, explored):
        walkable = []
        for r in range(self.env.size):
            for c in range(self.env.size):
                if self.env.base_grid[r][c] != CELL_WALL and not explored[r][c]:
                    walkable.append((r,c))
        return walkable
    
    def _walkable(self, r, c):
        if not (0 <= r < self.env.size and 0 <= c < self.env.size):
            return False
        return self.env.base_grid[r][c] != CELL_WALL
    
    def _bfs_dist(self, src, dst):
        if src == dst:
            return 0
        key = (src, dst)
        if key in self._dist_cache:
            return self._dist_cache[key]
        queue = deque([(src, 0)])
        visited = {src}
        while queue:
            (r, c), d = queue.popleft()
            for dr, dc in ACTIONS.values():
                nr, nc = r+dr, c+dc
                if (nr, nc) == dst:
                    self._dist_cache[key] = d+1
                    return d+1
                if (0<= nr < self.env.size and 0 <= nc < self.env.size and self.env.base_grid[nr][nc] != CELL_WALL and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), d+1))
        self._dist_cache[key] = 999 # if we can't reach the destination, return a large distance - this can happen if we're completely blocked by monsters, but it prevents crashing and allows the agent to at least try to find a way around
        return 999
    
    def _nearest_unexplored_in(self, pos, explored):
        # added to help with exploration - finds the nearest unexplored cell in the observed grid using BFS - recommended by Claude
        queue = deque([(pos, 0)])
        visited = {pos}
        while queue:
            (r, c), d = queue.popleft()
            if not explored[r][c]:
                return (r, c)
            for dr, dc in ACTIONS.values():
                nr, nc = r+dr, c+dc
                if (0<= nr < self.env.size and 0 <= nc < self.env.size and self.env.base_grid[nr][nc] != CELL_WALL and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), d+1))
        return None

