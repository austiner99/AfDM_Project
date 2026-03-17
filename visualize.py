# visualize.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# One color per cell type — swap these for sprite images later
CELL_COLORS = {
    0: '#2b2b2b',  # empty floor — dark gray
    1: '#4fc3f7',  # agent — blue
    2: '#81c784',  # exit — green
    3: '#ffd54f',  # treasure — gold
    4: '#e57373',  # monster — red
    5: '#5d4037',  # wall — brown
}

CELL_LABELS = {
    0: 'Floor',
    1: 'Agent',
    2: 'Exit',
    3: 'Treasure',
    4: 'Monster',
    5: 'Wall',
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

def visualize_environment(grid, title='Dungeon Environment'):
    fig, ax = plt.subplots(figsize=(8, 8))

    rgb = grid_to_rgb(grid)
    ax.imshow(rgb, origin='upper', interpolation='nearest')
    # interpolation='nearest' keeps pixels sharp — important for sprites later

    ax.set_title(title)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Discrete legend instead of colorbar
    present_vals = np.unique(grid)
    patches = [
        mpatches.Patch(color=CELL_COLORS[v], label=CELL_LABELS[v])
        for v in present_vals if v in CELL_COLORS
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=9,
              framealpha=0.8, borderpad=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from dungeon_env import DungeonEnv
    env = DungeonEnv(size=25, num_treasures=4, num_monsters=2)
    visualize_environment(env.grid)