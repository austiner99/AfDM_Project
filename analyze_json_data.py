#code to analyze data in the json file (treasure, steps, reward, time, win rate) and produce necessary graphs

#graphs wanted:
# 1a. box and whisker plot of average reward per episode for each agent
# 1b. box and whisker plot of steps taken per episode for each agent
# 1c. box and whisker plot of treasure held at end of episode for each agent
# 1d. box and whisker plot of time taken per episode for each agent
# 1. each of the above four plots should be separate figures, but each should have four subplots for game size (12, 18, 24, 30) (mdp did no size 30 runs)
#    subplots should all be horizontally aligned, with a shared y-axis and a common x-axis label of "Agent" and subtitle corresponding to the size (e.g. "Dungeon Size: 12x12")

# much of the graphing code was generated with claude and chatGPT help, and evaluated/edited by me.

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ── Load data ─────────────────────────────────────────────────────────────────

with open('/home/austiner99/classes/Algorithms_for_Decision_Making/AfDM_Project/experiment_results.json') as f:
    raw = json.load(f)

AGENTS      = ['mdp', 'mcts', 'pomcp']
AGENT_LABELS= ['MDP', 'MCTS', 'POMCP']
ALL_SIZES   = ['12', '18', '24', '30']
METRICS     = ['treasure', 'steps', 'reward', 'time']
METRIC_LABELS = {
    'treasure': 'Treasure Collected',
    'steps':    'Steps Taken',
    'reward':   'Total Reward',
    'time':     'Time (seconds)',
}
TITLES = {
    'treasure': 'Treasure Collected per Episode',
    'steps':    'Steps Taken per Episode',
    'reward':   'Total Reward per Episode',
    'time':     'Computation Time per Episode',
}

# Extract per-episode data indexed by [agent][size][metric_index]
# episodes format: [treasure, steps, reward, time]
METRIC_IDX = {'treasure': 0, 'steps': 1, 'reward': 2, 'time': 3}

def remove_outliers_iqr(data, k=1.5):
    """Return data with Tukey IQR outliers removed."""
    if data is None or len(data) == 0:
        return data
    x = np.asarray(data, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return x.tolist()  # nothing to filter meaningfully
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return x[(x >= lo) & (x <= hi)].tolist()

def get_data(agent, size, metric):
    if agent not in raw or size not in raw[agent]:
        return None
    eps = raw[agent][size]['episodes']
    idx = METRIC_IDX[metric]
    return [ep[idx] for ep in eps]

# ── Style ─────────────────────────────────────────────────────────────────────

AGENT_COLORS = {
    'mdp':   '#4e79a7',   # steel blue
    'mcts':  '#f28e2b',   # amber
    'pomcp': '#59a14f',   # sage green
}
AGENT_MEDIANCOLORS = {
    'mdp':   '#1a3f6b',
    'mcts':  '#9b5800',
    'pomcp': '#2d5a28',
}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.25,
    'grid.linestyle':   '--',
    'axes.labelsize':   10,
    'axes.titlesize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'figure.dpi':       150,
})

# ── Plotting helper ───────────────────────────────────────────────────────────

def make_figure(metric, data_filter=None, showfliers=True, title_suffix="", agent_filter=None):
    sizes_to_plot = ALL_SIZES
    n_cols = len(sizes_to_plot)

    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5.5), sharey=True)
    fig.suptitle(TITLES[metric] + title_suffix, fontsize=14, fontweight='bold', y=1.01)

    for col, size in enumerate(sizes_to_plot):
        ax = axes[col]

        # Which agents have data for this size?
        agents_here = [a for a in AGENTS if get_data(a, size, metric) is not None]

        # Apply optional per-subplot agent filtering
        if agent_filter is not None:
            agents_here = [a for a in agents_here if agent_filter(a, size, metric)]

        all_data = []
        for a in agents_here:
            d = get_data(a, size, metric)
            if data_filter is not None:
                d = data_filter(d)
            all_data.append(d)

        positions  = list(range(1, len(agents_here) + 1))
        labels     = [a.upper() for a in agents_here]
        colors     = [AGENT_COLORS[a] for a in agents_here]
        mcolors    = [AGENT_MEDIANCOLORS[a] for a in agents_here]

        bp = ax.boxplot(
            all_data,
            positions=positions,
            patch_artist=True,
            widths=0.55,
            notch=False,
            showfliers=showfliers,
            flierprops=dict(marker='o', markersize=3, alpha=0.4, linestyle='none'),
            medianprops=dict(linewidth=2.5),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            boxprops=dict(linewidth=1.2),
        )

        for patch, fc in zip(bp['boxes'], colors):
            patch.set_facecolor(fc)
            patch.set_alpha(0.75)

        if showfliers and 'fliers' in bp:
            for flier, fc in zip(bp['fliers'], colors):
                flier.set(markerfacecolor=fc, markeredgecolor=fc)

        for median, mc in zip(bp['medians'], mcolors):
            median.set_color(mc)

        for i, (data, fc) in enumerate(zip(all_data, colors), 1):
            jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(data))
            ax.scatter([i + j for j in jitter], data, alpha=0.2, s=10, color=fc, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel('Agent', fontsize=9)
        ax.set_title(f'Dungeon Size: {size}×{size}', fontsize=10, pad=6)
        ax.set_xlim(0.3, len(agents_here) + 0.7)

        # Existing label for size 30 (optional)
        if size == '30' and 'mdp' not in agents_here:
            ax.text(0.5, 0.97, 'MDP not evaluated\nat this size',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=7.5, color='#888888', style='italic')

        # NEW: optional label for "mdp intentionally excluded on 24"
        if metric == 'time' and size == '24' and 'mdp' not in agents_here:
            ax.text(0.5, 0.08, 'MDP excluded for this subplot',
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=7.5, color='#888888', style='italic')

    axes[0].set_ylabel(METRIC_LABELS[metric], fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig
# ── Generate all four figures ─────────────────────────────────────────────────

output_paths = {}
# Original plots (unchanged behavior)
for metric in METRICS:
    fig = make_figure(metric)
    path = f'/home/austiner99/classes/Algorithms_for_Decision_Making/AfDM_Project/outputs/plot2_{metric}.png'
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')

# Extra: time plot with outliers removed
fig = make_figure(
    'time',
    data_filter=lambda d: remove_outliers_iqr(d, k=1.5),
    showfliers=False,
    title_suffix=" (IQR outliers removed)"
)
path = '/home/austiner99/classes/Algorithms_for_Decision_Making/AfDM_Project/outputs/plot2_time_no_outliers.png'
fig.savefig(path, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'Saved: {path}')

#Extra: time plot with no mdp on the 24 size (since it had so many extreme outliers that the other agents were hard to see)
def ignore_mdp_on_24(agent, size, metric):
    return not (metric == 'time' and size == '24' and agent == 'mdp')

fig = make_figure(
    'time',
    title_suffix=" (MDP excluded at 24×24)",
    agent_filter=ignore_mdp_on_24,
    showfliers=True,      # or False if you also want to hide fliers
)

path = '/home/austiner99/classes/Algorithms_for_Decision_Making/AfDM_Project/outputs/plot_time_no_mdp_size24.png'
fig.savefig(path, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'Saved: {path}')

print('All done.')