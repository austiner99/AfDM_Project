#code that takes in the win rate from each agent on each size and produces a 3 line graphs on one figure, showing change in win rate across sizes for each agent. x-axis is size, y-axis is win rate, and there is a line for each agent. The graph should have a title, axis labels, and a legend.
#json structure: {"agent": {"size": {"stats": {"win_rate": value}}}}} saved to experiment_results.json (3 sizes for mdp, 4 sizes for mcts and pomcp)

import json

import matplotlib.pyplot as plt

with open('/home/austiner99/classes/Algorithms_for_Decision_Making/AfDM_Project/experiment_results.json') as f:
    raw = json.load(f)
AGENTS = ['mdp', 'mcts', 'pomcp']
AGENT_LABELS = ['MDP', 'MCTS', 'POMCP']
ALL_SIZES = ['12', '18', '24', '30']

def get_win_rate(agent, size):
    if agent not in raw or size not in raw[agent]:
        return None
    return raw[agent][size]['stats']['win_rate']

plt.figure(figsize=(8, 6))
for agent, label in zip(AGENTS, AGENT_LABELS):
    sizes = []
    win_rates = []
    for size in ALL_SIZES:
        win_rate = get_win_rate(agent, size)
        if win_rate is not None:
            sizes.append(int(size))
            win_rates.append(win_rate)
    plt.plot(sizes, win_rates, marker='o', label=label)
plt.title('Win Rate vs Dungeon Size for Each Agent')
plt.xlabel('Dungeon Size (NxN)')
plt.ylabel('Win Rate')
plt.xticks([12, 18, 24, 30])
plt.legend()
plt.grid()
plt.savefig('win_rate_comparison.png')
plt.show()