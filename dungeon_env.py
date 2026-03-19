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
MONSTER_DAMAGE_BOUNDRIES = (10, 25)
AGENT_ATTACK_DAMAGE = (15, 30)
STARTING_HP = 100
TREASURE_REWARD_BOUNDRIES = (50, 500)
VISION_RADIUS = 2
MONSTER_VISION_RADIUS = 5
MONSTER_NOTICE_CHANCE = 0.8
MONSTER_LOST_SIGHT_LIMIT = 4

class Monster:
    def __init__(self, pos):
        self.pos = pos
        self.hp = random.randint(20, 75)
        self.reward = self.hp
        self.alert = False
        self.lost_sight_steps = 0

    @property
    def alive(self):
        return self.hp > 0

class DungeonEnv:
    def __init__(self, size=25, num_treasures=4, num_monsters=2,
                 vision_radius=VISION_RADIUS,
                 monster_vision_radius=MONSTER_VISION_RADIUS,
                 monster_notice_chance=MONSTER_NOTICE_CHANCE):
        self.size                  = size
        self.num_treasures         = num_treasures
        self.num_monsters          = num_monsters
        self.vision_radius         = vision_radius
        self.monster_vision_radius = monster_vision_radius
        self.monster_notice_chance = monster_notice_chance
        self.agent_hp              = STARTING_HP
        self.treasure_held         = 0
        self.done                  = False
        self.message               = ""
        self._attacked_this_turn = set()
        self._generate()

    def _generate(self):
        self.agent_hp      = STARTING_HP
        self.treasure_held = 0
        self.done          = False
        self.message       = ""
        self._attacked_this_turn = set()
        self.grid          = np.zeros((self.size, self.size), dtype=int)
        self.explored      = np.zeros((self.size, self.size), dtype=bool)

        self.placed_rooms      = self._create_rooms()
        walkable               = self._walkable_cells()
        random.shuffle(walkable)

        self.agent_pos          = walkable.pop()
        self.exit_pos           = walkable.pop()
        self.treasure_positions = [walkable.pop() for _ in range(min(self.num_treasures, len(walkable)))]
        self.monsters           = [Monster(walkable.pop()) for _ in range(min(self.num_monsters, len(walkable)))]

        self._redraw_grid()
        self._update_vision()

    def reset(self):
        self._generate()
        return self.get_state()

    # ── Grid drawing ──────────────────────────────────────────────────────────

    def _redraw_grid(self):
        """
        Rebuild the display grid from scratch each time.
        This is the fix for the disappearing exit / treasure bug — instead of
        surgically clearing and stamping individual cells (which loses permanent
        features when entities move over them), we just redraw everything from
        the authoritative lists every step. Fast enough for any reasonable size.
        """
        # Start from the structural grid (walls + corridors carved during generation)
        self.grid = self.base_grid.copy()

        # Stamp permanent features
        self.grid[self.exit_pos] = CELL_EXIT
        for pos in self.treasure_positions:
            self.grid[pos] = CELL_TREASURE

        # Stamp moving entities on top
        for monster in self.monsters:
            if monster.alive:
                self.grid[monster.pos] = CELL_MONSTER

        # Agent drawn last so it's always visible even on exit/treasure cells
        self.grid[self.agent_pos] = CELL_AGENT

    def _create_rooms(self):
        """Build rooms and corridors, saving a clean structural copy as base_grid."""
        num_rooms    = random.randint(self.size // 4, max(self.size // 4 + 1, int(self.size // 1.5)))
        placed_rooms = []

        for _ in range(num_rooms):
            for _ in range(60):
                room_w = random.randint(4, self.size // 3)
                room_h = random.randint(4, self.size // 3)
                rx     = random.randint(1, self.size - room_w - 2)
                ry     = random.randint(1, self.size - room_h - 2)

                overlap = False
                for (ex, ey, ew, eh) in placed_rooms:
                    if not (rx + room_w + 1 <= ex or rx >= ex + ew + 1 or
                            ry + room_h + 1 <= ey or ry >= ey + eh + 1):
                        overlap = True
                        break

                if not overlap:
                    self.grid[rx:rx+room_w, ry:ry+room_h]             = CELL_WALL
                    self.grid[rx+1:rx+room_w-1, ry+1:ry+room_h-1]    = CELL_EMPTY
                    placed_rooms.append((rx, ry, room_w, room_h))
                    break

        for i in range(1, len(placed_rooms)):
            self._connect_rooms(placed_rooms[i-1], placed_rooms[i])

        # Save a clean copy that only has walls and floor — no entities
        self.base_grid = self.grid.copy()
        return placed_rooms

    def _connect_rooms(self, room_a, room_b):
        ax = room_a[0] + room_a[2] // 2
        ay = room_a[1] + room_a[3] // 2
        bx = room_b[0] + room_b[2] // 2
        by = room_b[1] + room_b[3] // 2
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

    # ── Vision ────────────────────────────────────────────────────────────────

    def _update_vision(self):
        ar, ac = self.agent_pos
        r_min = max(0, ar - self.vision_radius)
        r_max = min(self.size, ar + self.vision_radius + 1)
        c_min = max(0, ac - self.vision_radius)
        c_max = min(self.size, ac + self.vision_radius + 1)
        self.explored[r_min:r_max, c_min:c_max] = True

    def get_observed_grid(self):
        observed = self.grid.copy()
        observed[~self.explored] = -1
        return observed
    
    # -- Adjacency helpers ---------------------------------------------------
    def _adjacent_positions(self, pos):
        r, c = pos
        return [
            (r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
            if 0 <= r+dr < self.size and 0 <= c+dc < self.size
        ]
    
    def _adjacent_monsters(self, pos):
        adj = set(self._adjacent_positions(pos))
        return [m for m in self.monsters if m.alive and m.pos in adj]
    
    # -- Combat ----------------------------------------------------------------
    def _monsters_attack(self):
        self._attacked_this_turn = set()
        reward = 0
        messages = []
        for monster in self.monsters:
            if not monster.alive:
                continue
            if monster.pos in set(self._adjacent_positions(self.agent_pos)):
                damage = random.randint(*MONSTER_DAMAGE_BOUNDRIES)
                self.agent_hp -= damage
                reward -= damage
                messages.append(f"A monster attacks! -{damage} HP. ({self.agent_hp} HP remaining)")
                self._attacked_this_turn.add(id(monster))
                monster.alert = True  # Attacking also puts monster on alert
                monster.lost_sight_steps = 0
        return reward, " ".join(messages)
    
    def _agent_attack(self):
        targets = self._adjacent_monsters(self.agent_pos)
        if not targets:
            return 0, "No adjacent mosnters to attack."
        reward = 0
        messages = []

        for monster in targets:
            damage_agent = random.randint(*AGENT_ATTACK_DAMAGE)
            monster.hp -= damage_agent
            messages.append(f"\nYou attack a monster for {damage_agent} damage!")
            if not monster.alive:
                reward += monster.reward
                messages.append(f"You killed a monster! +{monster.reward} reward.")
        
        self.monsters = [m for m in self.monsters if m.alive]
        self.treasure_held += reward
        return reward, " ".join(messages)

    # ── Monster logic ─────────────────────────────────────────────────────────

    def _has_line_of_sight(self, monster_pos):
        mr, mc = monster_pos
        ar, ac = self.agent_pos
        same_row = (mr == ar)
        same_col = (mc == ac)
        if not same_row and not same_col:
            return False
        if same_row:
            if abs(ac - mc) > self.monster_vision_radius:
                return False
            step = 1 if ac > mc else -1
            for c in range(mc + step, ac, step):
                if self.base_grid[mr][c] == CELL_WALL:
                    return False
            return True
        if same_col:
            if abs(ar - mr) > self.monster_vision_radius:
                return False
            step = 1 if ar > mr else -1
            for r in range(mr + step, ar, step):
                if self.base_grid[r][mc] == CELL_WALL:
                    return False
            return True
        return False

    def _move_monsters(self):
        ar, ac = self.agent_pos
        occupied = {m.pos for m in self.monsters if m.alive}

        for monster in self.monsters:
            if not monster.alive:
                continue
            if id(monster) in self._attacked_this_turn:
                continue
            can_see = self._has_line_of_sight(monster.pos)

            if can_see:
                # Agent spotted — try to go alert (if not already)
                if not monster.alert:
                    if random.random() < self.monster_notice_chance:
                        monster.alert = True
                # Reset the lost-sight counter whenever we CAN see the agent
                monster.lost_sight_steps = 0
            else:
                # Can't see agent this step
                if monster.alert:
                    monster.lost_sight_steps += 1
                    # Only go idle after MONSTER_LOST_SIGHT_LIMIT consecutive
                    # steps without line of sight — not just one step
                    if monster.lost_sight_steps >= MONSTER_LOST_SIGHT_LIMIT:
                        monster.alert = False
                        monster.lost_sight_steps = 0
            if not monster.alert:
                continue

            if monster.alert:
                mr, mc = monster.pos
                dr = 0 if ar == mr else (1 if ar > mr else -1)
                dc = 0 if ac == mc else (1 if ac > mc else -1)
                candidates = []
                if dr != 0:
                    candidates.append((mr + dr, mc))
                if dc != 0:
                    candidates.append((mr, mc + dc))
                candidates.append((mr, mc))  # also consider standing still if blocked
                    
                for nr, nc in candidates:
                    if self._is_walkable(nr, nc) and (nr, nc) not in occupied and (nr, nc) != self.agent_pos:
                        monster.pos = (nr, nc)
                        break

            occupied.add(monster.pos)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _walkable_cells(self):
        return [(r, c) for r in range(self.size)
                       for c in range(self.size)
                       if self.grid[r][c] == CELL_EMPTY]

    def _is_walkable(self, r, c):
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        return self.base_grid[r][c] != CELL_WALL

    def get_state(self):
        return (
            self.agent_pos,
            self.exit_pos,
            self.treasure_positions,
            [m.pos for m in self.monsters],
        )

    def _info(self):
        return {
            'agent_hp':      self.agent_hp,
            'treasure_held': self.treasure_held,
            'message':       self.message,
            'done':          self.done,
        }

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, self._info()

        reward = -0.1
        self.message = ""

        mon_atk_reward, mon_atk_msg = self._monsters_attack()
        reward += mon_atk_reward
        if mon_atk_msg:
            self.message += mon_atk_msg + " "

        # Exit action
        if action == 'exit':
            if self.agent_pos == self.exit_pos:
                if self.treasure_held > 0:
                    reward += 50 + self.treasure_held
                    self.message = f"Exited with {self.treasure_held} treasure! You win!"
                else:
                    self.message = "Exited without treasure. No reward."
                self.done = True
                self._redraw_grid()
                return self.get_state(), reward, self.done, self._info()
            else:
                self.message = "Not at the exit."
                return self.get_state(), reward, self.done, self._info()
        elif action == 'attack':
            atk_reward, atk_msg = self._agent_attack()
            reward += atk_reward
            if atk_msg:
                self.message += atk_msg + " "
            
        # Movement
        elif action in ACTIONS:

            dr, dc = ACTIONS[action]
            new_r  = self.agent_pos[0] + dr
            new_c  = self.agent_pos[1] + dc
            monster_cells = {m.pos for m in self.monsters if m.alive}

            if not self._is_walkable(new_r, new_c):
                self.message = "Bumped into a wall."
            elif (new_r, new_c) in monster_cells:
                self.message = "Can't move onto a monster! Try attacking instead."
            else:
                self.agent_pos = (new_r, new_c)

                if self.agent_pos in self.treasure_positions:
                    gained = random.randint(*TREASURE_REWARD_BOUNDRIES)
                    self.treasure_held += gained
                    self.treasure_positions.remove(self.agent_pos)
                    reward += gained
                    self.message = f"Picked up {gained} treasure! Carrying {self.treasure_held}."
        else:
            self.message = "Unknown action."
        
        self._move_monsters()
        if self.agent_hp <= 0:
            reward -= 100
            self.done = True
            self.treasure_held = 0
            self.message += "You died. Womp Womp."

        self._update_vision()

        self._redraw_grid()
        return self.get_state(), reward, self.done, self._info()