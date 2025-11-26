# """
# Training script for Value Iteration and Q-Iteration.
# """

# import numpy as np
# import argparse
# import json
# import os
# from environment import GridWorldEnv
# from value_iteration import ValueIteration
# from q_iteration import QIteration


# def main():
#     """
#     Run both algorithms and save results.
#     """
#     parser = argparse.ArgumentParser(description='Train RL algorithms on GridWorld')
#     parser.add_argument('--seed', type=int, default=641, help='Random seed')
#     parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
#     parser.add_argument('--epsilon', type=float, default=1e-4, help='Convergence threshold')
#     parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations')
#     args = parser.parse_args()

#     # Create results directory
#     os.makedirs('results', exist_ok=True)
#     os.makedirs('results/visualizations', exist_ok=True)

#     # TODO: Initialize environment with seed
#     # TODO: Run Value Iteration
#     #       - Create ValueIteration solver
#     #       - Solve for optimal values
#     #       - Extract policy
#     #       - Save results
#     # TODO: Run Q-Iteration
#     #       - Create QIteration solver
#     #       - Solve for optimal Q-values
#     #       - Extract policy and values
#     #       - Save results
#     # TODO: Compare algorithms
#     #       - Print convergence statistics
#     #       - Check if policies match
#     #       - Save comparison results

#     raise NotImplementedError


# if __name__ == '__main__':
#     main()



"""
Training script for Value Iteration and Q-Iteration.
"""

import numpy as np
import argparse
import json
import os
from environment import GridWorldEnv
from value_iteration import ValueIteration
from q_iteration import QIteration
from typing import Dict, Any

# Define paths for saving results
RESULTS_DIR = 'results'
VI_RESULTS_FILE = os.path.join(RESULTS_DIR, 'value_iteration_results.json')
QI_RESULTS_FILE = os.path.join(RESULTS_DIR, 'q_iteration_results.json')
COMPARISON_FILE = os.path.join(RESULTS_DIR, 'comparison_summary.json')
VI_POLICY_FILE = os.path.join(RESULTS_DIR, 'vi_optimal_policy.npy')
QI_POLICY_FILE = os.path.join(RESULTS_DIR, 'qi_optimal_policy.npy')


def run_value_iteration(env: GridWorldEnv, args: argparse.Namespace) -> Dict[str, Any]:
    """Runs Value Iteration and saves results."""
    print("--- Running Value Iteration ---")
    
    vi_solver = ValueIteration(env, gamma=args.gamma, epsilon=args.epsilon)
    
    # Solve for optimal values
    V_star, n_iterations = vi_solver.solve(max_iterations=args.max_iter)
    
    # Extract optimal policy
    pi_star = vi_solver.extract_policy(V_star)
    
    # Save policy array
    np.save(VI_POLICY_FILE, pi_star)

    # Prepare results
    results = {
        'algorithm': 'Value Iteration',
        'n_iterations': n_iterations,
        'convergence_epsilon': args.epsilon,
        'discount_factor': args.gamma,
        'optimal_value_function_max': np.max(V_star).item(),
        'optimal_value_function_min': np.min(V_star).item(),
        # Store as lists for JSON serialization
        'optimal_value_function': V_star.tolist(),
        'optimal_policy': pi_star.tolist()
    }
    
    print(f"VI converged in {n_iterations} iterations.")
    print(f"Max optimal value V*(s): {results['optimal_value_function_max']:.4f}")
    
    with open(VI_RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

def run_q_iteration(env: GridWorldEnv, args: argparse.Namespace) -> Dict[str, Any]:
    """Runs Q-Iteration and saves results."""
    print("\n--- Running Q-Iteration ---")
    
    qi_solver = QIteration(env, gamma=args.gamma, epsilon=args.epsilon)
    
    # Solve for optimal Q-values
    Q_star, n_iterations = qi_solver.solve(max_iterations=args.max_iter)
    
    # Extract optimal policy and values
    pi_star = qi_solver.extract_policy(Q_star)
    V_star = qi_solver.extract_values(Q_star)
    
    # Save policy array
    np.save(QI_POLICY_FILE, pi_star)

    # Prepare results
    results = {
        'algorithm': 'Q-Iteration',
        'n_iterations': n_iterations,
        'convergence_epsilon': args.epsilon,
        'discount_factor': args.gamma,
        'optimal_value_function_max': np.max(V_star).item(),
        'optimal_value_function_min': np.min(V_star).item(),
        # Store as lists for JSON serialization
        'optimal_value_function': V_star.tolist(),
        'optimal_policy': pi_star.tolist(),
        'optimal_q_function': Q_star.tolist()
    }
    
    print(f"QI converged in {n_iterations} iterations.")
    print(f"Max optimal value V*(s): {results['optimal_value_function_max']:.4f}")
    
    with open(QI_RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

def compare_algorithms(vi_results: Dict[str, Any], qi_results: Dict[str, Any]):
    """Compares the results of the two algorithms."""
    print("\n--- Comparing Results ---")

    # Load policies (we saved them as numpy arrays)
    vi_policy = np.array(vi_results['optimal_policy'])
    qi_policy = np.array(qi_results['optimal_policy'])

    # Compare policies (ignoring terminal state if marked as -1)
    # Filter out the terminal state (4, 4) if applicable, but for comparison, a direct check works:
    policy_match = np.all(vi_policy == qi_policy)

    # Compare convergence speed
    vi_iter = vi_results['n_iterations']
    qi_iter = qi_results['n_iterations']

    comparison_summary = {
        'Value Iteration Iterations': vi_iter,
        'Q-Iteration Iterations': qi_iter,
        'Policies Match': bool(policy_match),
        'VI_V_max': vi_results['optimal_value_function_max'],
        'QI_V_max': qi_results['optimal_value_function_max'],
        'V_max_difference': np.abs(vi_results['optimal_value_function_max'] - qi_results['optimal_value_function_max']),
    }
    
    print(f"VI Iterations: {vi_iter}")
    print(f"QI Iterations: {qi_iter}")
    print(f"Policies Match: {policy_match}")

    with open(COMPARISON_FILE, 'w') as f:
        json.dump(comparison_summary, f, indent=4)


def main():
    """
    Run both algorithms and save results.
    """
    parser = argparse.ArgumentParser(description='Train RL algorithms on GridWorld')
    parser.add_argument('--seed', type=int, default=641, help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Convergence threshold')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations')
    args = parser.parse_args()

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'visualizations'), exist_ok=True)

    # Initialize environment with seed
    env = GridWorldEnv(seed=args.seed)

    # Run Value Iteration
    vi_results = run_value_iteration(env, args)
    
    # Run Q-Iteration
    qi_results = run_q_iteration(env, args)

    # Compare algorithms
    compare_algorithms(vi_results, qi_results)

    print("\nTraining complete. Results saved in the 'results' directory.")


if __name__ == '__main__':
    main()