import os
import numpy as np
from env.environment import Environment


def collect_expert_data(env_name, method):
    from student.get_data import sample_expert_data, sample_td3_data, sample_PPO_data, load_td3_model
    import args
    args = args.get_args()
    env = Environment(env_type=args.env_type)
    agent = load_td3_model(env,env_name)
    expert_data_path = f"expert_data/{env_name}_{method}.npy"
    os.makedirs(os.path.dirname(expert_data_path), exist_ok=True)

    if method == "astar":
        args.teacher = "Astar"
        _, states, actions = sample_expert_data(args, env, agent, batch_size=10)
    elif method == "rrt":
        args.teacher = "RRT"
        _, states, actions = sample_expert_data(args, env, agent, batch_size=10)
    elif method == "td3":
        args.teacher = "TD3"
        _, states, actions = sample_td3_data(args, env, agent, batch_size=10)
    print("states:", states)
    print("actions:", actions)
    np.save(expert_data_path, {'states': states, 'actions': actions})
    print(f"✅ {method.upper()} 数据已保存: {expert_data_path}")


if __name__ == "__main__":
    envs = ["env5"]
    methods = ["td3"]

    for env in envs:
        for method in methods:
            collect_expert_data(env, method)
