import argparse
from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from train import make_env

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate trained model for robot navigation')
    parser.add_argument('--model-path', type=str, default="models/td3.zip",
                       help='Path to the trained model')
    parser.add_argument('--num-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable visualization during evaluation')
    parser.add_argument('--vehicle-model', choices=['diff', '4wis'], default='diff',
                       help='Robot model abstraction used by the trained policy')
    args = parser.parse_args()

    # Load the trained TD3 model
    model = TD3.load(args.model_path)

    # Create the environment with visualization enabled unless disabled
    eval_env = DummyVecEnv([make_env(render=not args.no_render, vehicle_model=args.vehicle_model)])

    # Initialize metrics
    success_count = 0
    total_timesteps = 0
    total_distance = 0
    total_velocity = 0
    collision_count = 0
    time_limit_count = 0
    total_mode_switches = 0
    mode_ratio_sums = {
        'ackermann_ratio': 0.0,
        'lateral_ratio': 0.0,
        'parallel_ratio': 0.0,
    }
    
    # Run evaluation episodes
    for episode in range(args.num_episodes):
        # Get initial state
        obs = eval_env.reset()
        done = False
        total_reward = 0

        # Run episode
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)

            # Execute action
            obs, reward, done, info = eval_env.step(action)

            # Accumulate reward
            total_reward += reward[0]  # Get reward from first (and only) environment

        # Update metrics from episode info
        episode_info = info[0]  # Get info from first (and only) environment
        if episode_info['success']:
            success_count += 1
        if episode_info['collision']:
            collision_count += 1
        if episode_info['time_limit_reached']:
            time_limit_count += 1
            
        total_timesteps += episode_info['steps']
        total_distance += episode_info['total_distance']
        total_velocity += episode_info['average_velocity']
        total_mode_switches += episode_info.get('mode_switches', 0)
        for key in mode_ratio_sums:
            mode_ratio_sums[key] += episode_info.get(key, 0.0)

    # Calculate final metrics
    success_rate = (success_count / args.num_episodes) * 100
    collision_rate = (collision_count / args.num_episodes) * 100
    time_limit_rate = (time_limit_count / args.num_episodes) * 100
    avg_timesteps = total_timesteps / args.num_episodes
    avg_distance = total_distance / args.num_episodes
    avg_velocity = total_velocity / args.num_episodes
    avg_mode_switches = total_mode_switches / args.num_episodes

    # Print evaluation results
    print("\nFinal Evaluation Metrics:")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Collision Rate: {collision_rate:.1f}%")
    print(f"Time Limit Rate: {time_limit_rate:.1f}%")
    print(f"Average Timesteps per Episode: {avg_timesteps:.1f}")
    print(f"Average Distance Traveled: {avg_distance:.2f}")
    print(f"Average Velocity: {avg_velocity:.2f}")
    if args.vehicle_model == '4wis':
        print(f"Average Mode Switches: {avg_mode_switches:.1f}")
        print(f"Ackermann Usage: {(mode_ratio_sums['ackermann_ratio'] / args.num_episodes) * 100:.1f}%")
        print(f"Lateral Usage: {(mode_ratio_sums['lateral_ratio'] / args.num_episodes) * 100:.1f}%")
        print(f"Parallel Usage: {(mode_ratio_sums['parallel_ratio'] / args.num_episodes) * 100:.1f}%")
