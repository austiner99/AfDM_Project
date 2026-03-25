# DISCLAIMER: This was added at the end for human playing and debugging, because it was not a main part of the project, 
# most of this was generated with Claude code help, and evaluated/editted by me. 
# controls:
# w: up
# s: down
# a: left
# d: right
# q: quit 
# r: new game
# m: toggle cheat map (reveal dungeon)
# e: exit dungeon
# ?: help

from dungeon_env import DungeonEnv
from visualize import make_display, update_display
import random

KEY_MAP = {
    'w': 'up',  'up': 'up',
    's': 'down','down': 'down',
    'a': 'left','left': 'left',
    'd': 'right','right': 'right',
    'e': 'exit',
    'f': 'attack', 'attack': 'attack'
}

HELP = """
  w / up     — move north      s / down  — move south
  a / left   — move west       d / right — move east
  e          — exit dungeon (must be standing on the green exit cell)
  f          — attack (must be adjacent to a monster)
  m          — toggle cheat map (see full dungeon)
  r          — start a new game
  q          — quit
  ?          — show this help
"""

def play():
    env = DungeonEnv(size=20, num_treasures=random.randint(3, 8), num_monsters=random.randint(3, 8))
    info = env._info()
    info['message'] = 'Welcome to the Dungeon Crawler! Press ? for help.'
    cheat = False
    display = make_display(title='Dungeon Crawler')
    update_display(*display, env=env, info=info, cheat=cheat)

    print("\nDungeon Crawler — w/a/s/d move | e exit | f attack | m cheat | r restart | q quit | ? help")
    print("THe map window updates after each move. \n")

    while True:
        raw = input('Action: ').strip().lower()

        if not raw:
            continue
        if raw == 'q':
            print("Thanks for playing, quitter!")
            break
        if raw == '?':
            print(HELP)
            continue
        if raw == 'r':
            env = DungeonEnv(size=20, num_treasures=random.randint(3, 8), num_monsters=random.randint(3, 8))
            info = env._info()
            info['message'] = 'Game restarted! Press ? for help.'
            cheat = False
            update_display(*display, env=env, info=info, cheat=cheat)
            continue
        if raw == 'm':
            cheat = not cheat
            info = env._info()
            info['message'] = 'Cheat map ' + ('enabled' if cheat else 'disabled')
            update_display(*display, env=env, info=info, cheat=cheat)
            continue

        action = KEY_MAP.get(raw)
        if action is None:
            print(f"Unknown action '{raw}'. Press ? for help.")
            continue

        _, reward, done, info = env.step(action)
        update_display(*display, env=env, info=info, cheat=cheat)
        if info['message']:
            print(info['message'])
        if done:
            print("Game over! Press r to play again or q to quit.")

if __name__ == "__main__":
    play()
            