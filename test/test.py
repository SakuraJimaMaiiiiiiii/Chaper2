import numpy as np
import torch
from env.environment import Environment
from agent.agent import PPO, TD3
from args import get_test_args
import matplotlib.pyplot as plt
import os
from datetime import datetime
from train.train import PPOConfig
from matplotlib.patches import Rectangle


def load_ppo_model(env, model_path, args):
    """加载PPO模型"""
    # 创建PPO配置
    config = PPOConfig(args)
    agent = PPO(env, config)
    agent.ac.load_state_dict(torch.load(model_path))
    return agent

def load_td3_model(env, model_path):
 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = TD3(state_dim, action_dim, max_action)
    agent.load(model_path)
    return agent

def show_obs(ax, x1, y1, z1, x2, y2, z2):
 
 
    xx, yy = np.meshgrid([x1, x2], [y1, y2])
    
   
    ax.plot_surface(xx, yy, np.full_like(xx, z1), color='gray', alpha=0.3)
    ax.plot_surface(xx, yy, np.full_like(xx, z2), color='gray', alpha=0.3)
    

    yy, zz = np.meshgrid([y1, y2], [z1, z2])
    ax.plot_surface(np.full_like(yy, x1), yy, zz, color='gray', alpha=0.3)
    ax.plot_surface(np.full_like(yy, x2), yy, zz, color='gray', alpha=0.3)
    

    xx, zz = np.meshgrid([x1, x2], [z1, z2])
    ax.plot_surface(xx, np.full_like(xx, y1), zz, color='gray', alpha=0.3)
    ax.plot_surface(xx, np.full_like(xx, y2), zz, color='gray', alpha=0.3)

def show_obs2(ax, coords, plane='xy'):
 
    (x1, y1, z1), (x2, y2, z2) = coords
    
    if plane == 'xy':
        rect = Rectangle((x1, y1), x2-x1, y2-y1, facecolor='gray', alpha=0.3)
    elif plane == 'xz':
        rect = Rectangle((x1, z1), x2-x1, z2-z1, facecolor='gray', alpha=0.3)
    elif plane == 'yz':
        rect = Rectangle((y1, z1), y2-y1, z2-z1, facecolor='gray', alpha=0.3)
    ax.add_patch(rect)

def save(path_points, obstacles, env, args, episode_idx):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"test_results/{args.algorithm}_{args.env_type}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    path_points = np.array(path_points)
    

    fig = plt.figure(figsize=(20, 15))
    

    ax1 = fig.add_subplot(221, projection='3d')

    for (p1, p2) in obstacles:
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        show_obs(ax1, x1, y1, z1, x2, y2, z2)

    ax1.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 'b-', label='路径', linewidth=2)
    ax1.scatter(path_points[0, 0], path_points[0, 1], path_points[0, 2], 
               color='g', s=100, label='起点')
    ax1.scatter(path_points[-1, 0], path_points[-1, 1], path_points[-1, 2], 
               color='r', s=100, label='终点')
    ax1.set_xlabel('X轴')
    ax1.set_ylabel('Y轴')
    ax1.set_zlabel('Z轴')
    ax1.set_title('3D路径图')
    ax1.set_xlim(0, env.grid_size)
    ax1.set_ylim(0, env.grid_size)
    ax1.set_zlim(0, env.grid_size)
    ax1.legend()
    

    ax2 = fig.add_subplot(222)
   
    for coords in obstacles:
        show_obs2(ax2, coords, plane='xy')
    ax2.plot(path_points[:, 0], path_points[:, 1], 'b-', linewidth=2)
    ax2.scatter(path_points[0, 0], path_points[0, 1], color='g', s=100, label='起点')
    ax2.scatter(path_points[-1, 0], path_points[-1, 1], color='r', s=100, label='终点')
    ax2.set_xlabel('X轴')
    ax2.set_ylabel('Y轴')
    ax2.set_title('俯视图 (X-Y平面)')
    ax2.grid(True)
    ax2.set_xlim(0, env.grid_size)
    ax2.set_ylim(0, env.grid_size)
    ax2.legend()
    
    
    ax3 = fig.add_subplot(223)

    for coords in obstacles:
        show_obs2(ax3, coords, plane='xz')
    ax3.plot(path_points[:, 0], path_points[:, 2], 'b-', linewidth=2)
    ax3.scatter(path_points[0, 0], path_points[0, 2], color='g', s=100, label='起点')
    ax3.scatter(path_points[-1, 0], path_points[-1, 2], color='r', s=100, label='终点')
    ax3.set_xlabel('X轴')
    ax3.set_ylabel('Z轴')
    ax3.set_title('主视图 (X-Z平面)')
    ax3.grid(True)
    ax3.set_xlim(0, env.grid_size)
    ax3.set_ylim(0, env.grid_size)
    ax3.legend()
    

    ax4 = fig.add_subplot(224)

    for coords in obstacles:
        show_obs2(ax4, coords, plane='yz')
    ax4.plot(path_points[:, 1], path_points[:, 2], 'b-', linewidth=2)
    ax4.scatter(path_points[0, 1], path_points[0, 2], color='g', s=100, label='起点')
    ax4.scatter(path_points[-1, 1], path_points[-1, 2], color='r', s=100, label='终点')
    ax4.set_xlabel('Y轴')
    ax4.set_ylabel('Z轴')
    ax4.set_title('侧视图 (Y-Z平面)')
    ax4.grid(True)
    ax4.set_xlim(0, env.grid_size)
    ax4.set_ylim(0, env.grid_size)
    ax4.legend()
    
    plt.tight_layout()

    save_path = f"{save_dir}/result_{episode_idx+1}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


    return save_dir

def test_model():
   
    args = get_test_args()
    
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    

    env = Environment(render_mode=args.render, env_type=args.env_type)

    if args.algorithm == 'ppo':
        agent = load_ppo_model(env, args.model_path, args)
    else:
        agent = load_td3_model(env, args.model_path)
        
    print(f"\n{'='*20} 开始测试 {'='*20}")
    print(f"算法: {args.algorithm.upper()}")
    print(f"环境: {args.env_type}")
    print(f"模型路径: {args.model_path}")
    
    total_rewards = []
    total_steps = []
    success_count = 0
    
    for i in range(args.test_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        path_points = [env.position.copy()]  
        
        print(f"\n测试回合 {i+1}/{args.test_episodes}")
        
        while True:
            if args.render:
                env.render()
            
        
            if args.algorithm == 'ppo':
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                    action, _, _ = agent.ac(state_tensor)
                    action = action.cpu().numpy()
            else:
                action = agent.select_action(state)
            

            next_state, reward, done, info = env.step(action)
            path_points.append(env.position.copy()) 
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done:
                if info.get('distance_to_goal', 1.0) < env.delta/env.max_distance:
                    success_count += 1
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
   
        
        
        print(f"回合奖励: {episode_reward:.2f}")
        print(f"步数: {steps}")
        
    print(f"\n{'='*20} 测试结果 {'='*20}")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"平均步数: {np.mean(total_steps):.2f} ± {np.std(total_steps):.2f}")
    print(f"成功率: {success_count/args.test_episodes*100:.2f}%")
 
    env.close()

if __name__ == '__main__':
    test_model()





