import subprocess
import ast
import json
import numpy as np
import sys

AGENTS = ['mdp', 'mcts', 'pomcp']
SIZES = [12, 18, 24, 30]
EPISODES = 50
OUTPUT_FILE = "experiment_results.json"


def run_single(agent, size, episodes, seed):
    cmd = [
        sys.executable, "run_agent.py",
        "--agent", agent,
        "--episodes", str(episodes),
        "--size", str(size),
        "--delay", "0.0",
        "--no-display", 
        "--verbose"
    ]

    if seed is not None:
        cmd += ["--seed", str(seed)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"{agent} failed on size {size}")

    # Last line is the results list
    raw_list = result.stdout.strip().split("\n")[-1]
    return ast.literal_eval(raw_list)


def compute_stats(data):
    data = np.array(data)
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data))
    }


def print_stats(agent, size, results, stats):
    treasure = results["treasure"]
    steps = results["steps"]
    reward = results["reward"]
    time = results["time"]

    win_rate = stats["win_rate"]

    print(f"\n{'='*60}")
    print(f"Agent: {agent.upper()} | Size: {size}")
    print(f"{'-'*60}")
    print(f"Win Rate:        {win_rate*100:.1f}%")
    print(f"Treasure:        mean={stats['treasure']['mean']:.2f}, std={stats['treasure']['std']:.2f}")
    print(f"Steps:           mean={stats['steps']['mean']:.2f}, std={stats['steps']['std']:.2f}")
    print(f"Reward:          mean={stats['reward']['mean']:.2f}, std={stats['reward']['std']:.2f}")
    print(f"Time (sec):      mean={stats['time']['mean']:.2f}, std={stats['time']['std']:.2f}")
    print(f"{'='*60}\n")


def main():
    all_results = {}

    for agent in AGENTS:
        all_results[agent] = {}

        for size in SIZES:
            if agent == 'mdp' and size== 30:
                print(f"\nSkipping MDP agent on size {size} due to long runtime...")
                continue
            print(f"\nRunning {agent.upper()} on size {size}...")

            results = run_single(agent, size, EPISODES, seed=None)

            treasure = [r[0] for r in results]
            steps = [r[1] for r in results]
            reward = [r[2] for r in results]
            time = [r[3] for r in results]

            stats = {
                "treasure": compute_stats(treasure),
                "steps": compute_stats(steps),
                "reward": compute_stats(reward),
                "time": compute_stats(time),
                "win_rate": sum(1 for r in results if r[0] > 0) / len(results)
            }

            # Print results
            print_stats(agent, size, {
                "treasure": treasure,
                "steps": steps,
                "reward": reward,
                "time": time
            }, stats)

            # Save everything
            all_results[agent][size] = {
                "episodes": results,
                "treasure": treasure,
                "steps": steps,
                "reward": reward,
                "time": time,
                "stats": stats
            }

    # Save to file - this was added by ChatGPT to have a record of all results.
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()