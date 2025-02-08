from agent.td3 import TD3
from teacher.get_pair import get_pair
import numpy as np
import random


def load_td3_model(env, device='cpu'):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = TD3(state_dim, action_dim, max_action)

    model_path1 = r'..\finalresult\savemodel\env3\td3\Actor_net_step1000.pth'
    model_path2 = r'..\finalresult\savemodel\env3\td3\Critic1_net_step1000.pth'
    model_path3 = r'..\finalresult\savemodel\env3\td3\Critic2_net_step1000.pth'

    agent.load(model_path1, model_path2, model_path3, device)
    return agent


def sample_td3_data(args, env, agent, batch_size):
    states = []
    actions = []
    episodes = 1
    samples = 256
    for i in range(samples):
        for i in range(episodes):
            state = env.reset()
            while True:
                if args.render:
                    env.render()
                action = agent.select_action(state)
                states.append(state)
                actions.append(action)
                next_state, reward, done, info = env.step(action)
                state = next_state
                if done:
                    break
        env.close()
    assert batch_size <= len(states)
    index = random.sample(range(len(states)), batch_size)
    states = [states[i] for i in index]
    actions = [actions[i] for i in index]

    return index, np.array(states), np.array(actions)


# args = args.get_args()
# env = Environment(env_type=args.env_type)
# agent = load_td3_model(env)
# index, states, actions = sample_td3_data(env, agent, batch_size = 5)
# print(f"index:{index},actions{actions}")


# sample from RRT,Astar
def sample_expert_data(env, args, batch_size):
    pair, states, actions = get_pair(env, args)
    assert batch_size <= len(states)
    index = random.sample(range(len(states)), batch_size)
    states = [states[i] for i in index]
    actions = [actions[i] for i in index]
    return index, np.array(states), np.array(actions)

# args = args.get_args()
# env = Environment(env_type=args.env_type)
# index, states, actions = sample_expert_data(env, args, batch_size = 100)
# print(f"index:{index},actions{actions}")