import numpy as np
import random

CELL_EMPTY   = 0
CELL_WALL    = 5
CELL_AGENT   = 1
CELL_EXIT    = 2
CELL_TREASURE= 3
CELL_MONSTER = 4
ACTIONS = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
}
MONSTER_DAMAGE_BOUNDRIES = (10, 50)  # damage taken when stepping on a monster
STARTING_HP = 100
TREASURE_REWARD_BOUNDRIES = (50, 500)  # min and max reward for picking up a treasure
VISION_RADIUS = 3  # how far the agent can see in each direction

class DungeonEnv:
    def __init__(self, size=25, num_treasures=4, num_monsters=2):
        self.size         = size
        self.num_treasures= num_treasures
        self.num_monsters = num_monsters
        self.agent_hp     = STARTING_HP
        self.treasure_held= 0
        self.done         = False
        self._generate()

    def _generate(self):
        """Full dungeon generation in the correct order."""
        self.grid = np.zeros((self.size, self.size), dtype=int)

        # 1. Build rooms and corridors first
        self.placed_rooms = self._create_rooms()

        # 2. Now place entities on walkable (value 0) cells only
        walkable = self._walkable_cells()
        random.shuffle(walkable)

        self.agent_pos         = walkable.pop()
        self.exit_pos          = walkable.pop()
        self.treasure_positions= [walkable.pop() for _ in range(min(self.num_treasures, len(walkable)))]
        self.monster_positions = [walkable.pop() for _ in range(min(self.num_monsters,  len(walkable)))]

        # 3. Write entities onto grid
        self._update_grid()

    def reset(self):
        self._generate()
        return self.get_state()

    def _create_rooms(self):
        num_rooms   = random.randint(self.size // 4, self.size//1.5)
        placed_rooms= []

        for _ in range(num_rooms):
            for _ in range(60):  # max attempts per room
                room_w = random.randint(4, self.size // 3)
                room_h = random.randint(4, self.size // 3)
                rx = random.randint(1, self.size - room_w - 2)
                ry = random.randint(1, self.size - room_h - 2)

                # Check overlap (with 1-cell gap between rooms)
                overlap = False
                for (ex, ey, ew, eh) in placed_rooms:
                    if not (rx + room_w + 1 <= ex or rx >= ex + ew + 1 or
                            ry + room_h + 1 <= ey or ry >= ey + eh + 1):
                        overlap = True
                        break

                if not overlap:
                    # Draw walls on edges, floor (0) on interior
                    self.grid[rx:rx+room_w, ry:ry+room_h] = CELL_WALL     # fill with wall
                    self.grid[rx+1:rx+room_w-1, ry+1:ry+room_h-1] = CELL_EMPTY  # carve interior
                    placed_rooms.append((rx, ry, room_w, room_h))
                    break

        # Connect rooms with L-shaped corridors
        for i in range(1, len(placed_rooms)):
            self._connect_rooms(placed_rooms[i-1], placed_rooms[i])

        return placed_rooms

    def _connect_rooms(self, room_a, room_b):
        """Carve an L-shaped corridor between the centers of two rooms."""
        ax = room_a[0] + room_a[2] // 2
        ay = room_a[1] + room_a[3] // 2
        bx = room_b[0] + room_b[2] // 2
        by = room_b[1] + room_b[3] // 2

        # Horizontal then vertical (or vice versa randomly)
        if random.random() < 0.5:
            self._carve_h(ax, bx, ay)
            self._carve_v(ay, by, bx)
        else:
            self._carve_v(ay, by, ax)
            self._carve_h(ax, bx, by)

    def _carve_h(self, x1, x2, y):
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if 0 <= x < self.size and 0 <= y < self.size:
                self.grid[x][y] = CELL_EMPTY

    def _carve_v(self, y1, y2, x):
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if 0 <= x < self.size and 0 <= y < self.size:
                self.grid[x][y] = CELL_EMPTY

    def _walkable_cells(self):
        """Return all floor cells (value 0)."""
        return [(r, c) for r in range(self.size)
                       for c in range(self.size)
                       if self.grid[r][c] == CELL_EMPTY]

    def _update_grid(self):
        """Write entity positions onto the grid — no room generation here."""
        self.grid[self.agent_pos]  = CELL_AGENT
        self.grid[self.exit_pos]   = CELL_EXIT
        for pos in self.treasure_positions:
            self.grid[pos] = CELL_TREASURE
        for pos in self.monster_positions:
            self.grid[pos] = CELL_MONSTER

    def get_state(self):
        return self.agent_pos, self.exit_pos, self.treasure_positions, self.monster_positions
    
    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, self._info()
        
        reward = -0.1
        
        if action == 'exit':
            if self.agent_pos == self.exit_pos:
                if self.treasure_held > 0:
                    reward += 50 + self.treasure_held
                    self.message = f"Exited with {self.treasure_held} treasures! +{self.treasure_held} reward."
                else:
                    self.message = "Exited without treasures. No reward."
                self.done = True
                return self.get_state(), reward, self.done, self._info()
            else:
                self.message = "Tried to exit but not at the exit. No reward."
                return self.get_state(), reward, self.done, self._info()
        if action not in ACTIONS.values():
            self.message = "Invalid action. No reward."
            return self.get_state(), reward, self.done, self._info()
        
        dr, dc = ACTIONS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc
        
        if not self._is_walkable(new_r, new_c):
            self.message = "Bumped into a wall. No reward."
            return self.get_state(), reward, self.done, self._info()
        else:
            self.grid[self.agent_pos] = CELL_EMPTY
            self.agent_pos = (new_r, new_c) 
            
            if self.agent_pos in self.treasure_positions:
                self.treasure_held += random.randint(*TREASURE_REWARD_BOUNDRIES)
                self.treasure_positions.remove(self.agent_pos)
                reward += random.randint(*TREASURE_REWARD_BOUNDRIES)
                self.message = f"Picked up a treasure! Carrying {self.treasure_held} treasure."
                
            self.grid[self.agent_pos] = CELL_AGENT
        
        self._move_monsters()
        
        if self.agent_pos in self.monster_positions:
            damage = random.randint(*MONSTER_DAMAGE_BOUNDRIES)
            self.agent_hp -= damage
            reward -= damage
            self.message = f"Stepped on a monster! Lost {damage} health."
            
        if self.agent_hp <= 0:
            reward -= 100
            self.done = True
            self.message = "Died from monster damage. Game over."
            self.treasure_held = 0
            
        return self.get_state(), reward, self.done, self._info()
    
    def _is_walkable(self, r, c):
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        return self.grid[r][c] != CELL_WALL
    
    def _move_monsters(self):
        ar, ac = self.agent_pos
        new_monster_positions = []
        for pos in self.monster_positions:
            mr, mc = pos
            self.grid[pos] = CELL_EMPTY
            
            #decide direction towards agent
            dr = 0 if ar == mr else (1 if ar > mr else -1)
            dc = 0 if ac == mc else (1 if ac > mc else -1)
            
            candidates = [
                (mr+dr, mc),
                (mr, mc+dc),
                (mr, mc)
            ]
            moved = False
            for nr, nc in candidates:
                if self._is_walkable(nr, nc) and (nr, nc) not in new_monster_positions:
                    new_monster_positions.append((nr, nc))
                    moved = True
                    break
            if not moved:
                new_monster_positions.append(pos)
                
        self.monster_positions = new_monster_positions
        for pos in self.monster_positions:
            self.grid[pos] = CELL_MONSTER
            
    def _info(self):
        return {
            'agent_hp':     self.agent_hp,
            'treasure_held':self.treasure_held,
            'message':      self.message, 
            'done':         self.done
        }