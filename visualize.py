# visualize.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# One color per cell type — swap these for sprite images later
CELL_COLORS = {
    -1: "#595959",  # fog of war — light gray
     0: '#2b2b2b',  # floor — dark gray
     1: '#4fc3f7',  # agent — blue
     2: '#81c784',  # exit — green
     3: '#ffd54f',  # treasure — gold
     4: '#e57373',  # monster idle — red
     5: '#5d4037',  # wall — brown
     6: '#ff5252',  # monster alert — bright red
}

CELL_LABELS = {
    -1: 'Unexplored',
    0: 'Floor',
    1: 'Agent',
    2: 'Exit',
    3: 'Treasure',
    4: 'Monster',
    5: 'Wall',
    6: 'Monster (Alert)'
}

def grid_to_rgb(grid):
    """Convert integer grid to an RGB image array using CELL_COLORS."""
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cell_val, hex_color in CELL_COLORS.items():
        # Convert hex to RGB
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        mask = grid == cell_val
        rgb[mask] = (r, g, b)
    return rgb

def make_display(title='Dungeon Crawler'):
    plt.ion()  # interactive mode on
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    dummy = np.zeros((1, 1, 3), dtype=np.uint8)
    im = ax.imshow(dummy, origin='upper', interpolation='nearest')

    ax.set_title(title, color='white', fontsize=13, pad = 10)
    ax.set_xlabel('Column', color='#aaaaaa', fontsize=9)
    ax.set_ylabel('Row',    color='#aaaaaa', fontsize=9)
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
 
    # Status text below the grid (HP, gold, last message)
    txt = fig.text(0.5, 0.01, '', ha='center', va='bottom',
                   color='white', fontsize=10,
                   fontfamily='monospace',
                   bbox=dict(facecolor='#2a2a2a', edgecolor='#444444',
                             boxstyle='round,pad=0.4'))
 
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show(block=False)
    return fig, ax, im, txt

def update_display(fig, ax, im, txt, env, info, cheat=False):
    """
    Refresh the dungeon display in-place. Call this after every step.
 
    fig, ax, im, txt  — tuple from make_display()
    env               — the DungeonEnv instance
    info              — dict from env.step() or env._info()
    cheat             — True shows full map, False shows fog of war
    """
    grid = env.grid.copy() if cheat else env.get_observed_grid()
    alert_positions= {m.pos for m in env.monsters if m.alert}
    for (r, c) in alert_positions:
        if grid[r][c] == 4:  # if it's a monster cell, change to alert color
            grid[r][c] = 6
    rgb = grid_to_rgb(grid)
    im.set_data(rgb)
    im.set_extent([-0.5, grid.shape[1] - 0.5, 
                   grid.shape[0] - 0.5, -0.5])
    present_vals = np.unique(grid)
    patches = [
        mpatches.Patch(color=CELL_COLORS[v], label=CELL_LABELS[v])
        for v in present_vals if v in CELL_COLORS
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=8,
              framealpha=0.7, borderpad=0.4,
              facecolor='#2a2a2a', edgecolor='#444444',
              labelcolor='white')
    hp = info.get('agent_hp', '?')
    treasure = info.get('treasure_held', 0)
    mode = '[CHEAT]' if cheat else ''
    message = info.get('message', '')
    if message:
        mode += f" - {message}"
    txt.set_text(f"HP: {hp}/100, Treasure: {treasure} {mode}")

    if alert_positions:
        ax.set_title('Dungeon Crawler - !! Monster Alert !!',
                     color='#ff5252', fontsize=13, pad = 10)
    else:
        ax.set_title('Dungeon Crawler', color='white', fontsize=13, pad = 10)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)  # small pause to allow GUI to update

def visualize_environment(grid, title='Dungeon Environment'):
    """Utility to visualize a static grid (for debugging)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
 
    rgb = grid_to_rgb(grid)
    ax.imshow(rgb, origin='upper', interpolation='nearest')
    ax.set_title(title, color='white', fontsize=13)
    ax.set_xlabel('Column', color='#aaaaaa')
    ax.set_ylabel('Row',    color='#aaaaaa')
    ax.tick_params(colors='#aaaaaa')
 
    present_vals = np.unique(grid)
    patches = [
        mpatches.Patch(color=CELL_COLORS[v], label=CELL_LABELS[v])
        for v in present_vals if v in CELL_COLORS
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=9,
              framealpha=0.8, facecolor='#2a2a2a', labelcolor='white')
 
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from dungeon_env import DungeonEnv
    env = DungeonEnv(size=20, num_treasures=4, num_monsters=2)
    visualize_environment(env.grid, title='Full Map (MDP view)')
    visualize_environment(env.get_observed_grid(), title='Agent View (Explored Cells Only)')