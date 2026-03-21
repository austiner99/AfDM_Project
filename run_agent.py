#code to run the agent and watch the AI play the dungeon
# Usage (assisted by AI):
#   python run_agent.py --agent mdp
#   python run_agent.py --agent mcts
#   python run_agent.py --agent pomcp
#   python run_agent.py --agent mdp --episodes 10 --no-display
#   python run_agent.py --agent mdp --size 15 --monsters 3 --treasures 3
#   python run_agent.py --agent mdp --fog        # show fog-of-war view

import argparse
import time
import random

from dungeon_env import DungeonEnv
from visualize import make_display, update_display


def build_agent(name, env):
    if name == 'mdp':
        from mdp_agent import MDPAgent
        return MDPAgent(env)
    elif name == 'mcts':
        from mcts_agent import MCTSAgent
        return MCTSAgent(env)
    elif name == 'pomcp':
        from pomcp_agent import POMCPAgent
        return POMCPAgent(env)
    else:
        raise ValueError(f"Unknown agent name: {name}")
    
def run_episode(env, agent, display=None, delay = 0.15, verbose = True, cheat = True):
    obs = env.get_state()
    info = env._info()
    if display:
        update_display(*display, env=env, info=info, cheat=cheat)
        
    total_reward = 0.0
    steps = 0
    timeout = 200 #arbitrary timeout to prevent infinite loops
    while not info['done']:
        action = agent.act(obs, info)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if display:
            update_display(*display, env=env, info=info, cheat=cheat)
            time.sleep(delay)
            
        if verbose and info['message']:
            print(f"Step {steps:>3} [{action:>6}] - {info['message']} (Reward: {reward:+.1f}, Total: {total_reward:.1f})")
            
        if steps >= timeout:
            if verbose:
                print(f"Episode timed out after {steps} steps.")
            break
    
    return info['treasure_held'], steps, total_reward

def print_summary(results, agent_name, episodes):
    #following was formatted with chatGPT, feel free to change
    treasure = [r[0] for r in results]
    steps = [r[1] for r in results]
    rewards = [r[2] for r in results]
    wins = sum(1 for t in treasure if t > 0)
    print(f"\n{'='*50}")
    print(f"    Agent:              {agent_name.upper()}")
    print(f"    Episodes:           {episodes}")
    print(f"    Win rate:           {wins}/{episodes} ({wins / episodes * 100:.1f}%)")
    print(f"    Average Treasure:   {sum(treasure)/len(treasure):.1f} (max {max(treasure)})")
    print(f"    Average Steps:      {sum(steps) / len(steps):.1f}")
    print(f"    Average Reward:     {sum(rewards) / len(rewards):.1f} (max {max(rewards):.1f})")
    print(f"{'='*50}\n")
    
def main():
    #parser stuff was added and recommended by claude, feel free to change
    parser = argparse.ArgumentParser(description="Run an agent in the Dungeon Environment.")
    parser.add_argument('--agent', choices = ['mdp', 'mcts', 'pomcp'], default='mdp', help="Which agent to run")
    parser.add_argument('--episodes', type=int, default=1, help="Number of episodes to run (default: 1)")
    parser.add_argument('--size', type=int, default=20, help="Size of the dungeon grid (default: 20, must be within 12-30)")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility (default: None)")
    parser.add_argument('--delay', type=float, default=0.15, help="Delay between steps in seconds (default: 0.15)")
    parser.add_argument('--no-display', action='store_true', help="Disable visualization (for faster runs)")
    parser.add_argument('--cheat', action = 'store_true', default=True, help = "Whether to show the cheat map (full dungeon layout) in the visualization")
    parser.add_argument('--fog', action = 'store_true', default=False, help = "Whether to show the fog of war (only visible cells) in the visualization")
    parser.add_argument('--verbose', action='store_true', help="Print step-by-step info (default: False)")
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    cheat_view = not args.fog
    
    display = None
    if not args.no_display:
        print("Visualization disabled. Running in fast mode...")
        display = make_display(title=f"{args.agent.upper()} Agent - Fast Mode (No Visualization)")
    results = []
    for ep in range(args.episodes):
        env = DungeonEnv(size = args.size)
        agent = build_agent(args.agent, env)
        verbose = args.verbose or args.episodes == 1
        if args.episodes > 1:
            print(f"Episode {ep+1}/{args.episodes}", end=' - ', flush=True)
        gold, steps, reward = run_episode(env, agent, display=display, delay=args.delay, verbose=verbose, cheat=cheat_view)
        results.append((gold, steps, reward))
        if args.episodes > 1:
            outcome = "WIN " if gold > 0 else "LOSS"
            print(f"{outcome} - Treasure: {gold}, Steps: {steps}, Reward: {reward:.1f}")
    if args.episodes > 1:
        print_summary(results, args.agent, args.episodes)
    if display and args.episodes == 1:
        input ("\nPress Enter to close")
        
if __name__ == '__main__':
    main()