import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    # 通用参数
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'td3'], default='td3',
                      help ='选择训练算法(ppo或td3)')

    parser.add_argument('--env_type', type=str, choices=['env3', 'env4', 'env5'], 
                      default='env5', help='选择环境类型')
    parser.add_argument('--render', type=bool, default=False, help='是否渲染环境') 
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max_episodes', type=int, default=1000, help='最大训练回合数')

    parser.add_argument('--model_path', type=str, default='models/td3_env5', help='模型保存路径')


    # PPO特定参数
    parser.add_argument('--ppo_gamma', type=float, default=0.99, help='PPO折扣因子')
    parser.add_argument('--ppo_lam', type=float, default=0.95, help='GAE lambda参数')
    parser.add_argument('--ppo_clip_ratio', type=float, default=0.1, help='PPO剪切比例')
    parser.add_argument('--ppo_lr', type=float, default=3e-4, help='PPO学习率')
    parser.add_argument('--ppo_train_iters', type=int, default=80, help='每次更新的训练迭代次数')
    parser.add_argument('--ppo_target_kl', type=float, default=0.05, help='目标KL散度')
    parser.add_argument('--ppo_batch_size', type=int, default=5000, help='PPO batch大小')
    parser.add_argument('--ppo_minibatch_size', type=int, default=64, help='PPO minibatch大小')
    parser.add_argument('--ppo_max_ep_len', type=int, default=200, help='PPO最大回合长度')
    
    
    
    # TD3特定参数
    parser.add_argument('--td3_lr', type=float, default=3e-4, help='TD3学习率')
    parser.add_argument('--td3_batch_size', type=int, default=100, help='TD3 batch大小')
    parser.add_argument('--td3_warmup_steps', type=int, default=5000, help='预热步数')
    parser.add_argument('--td3_noise', type=float, default=0.1, help='探索噪声')
    parser.add_argument('--td3_noise_clip', type=float, default=0.5, help='噪声裁剪范围')
    parser.add_argument('--td3_policy_noise', type=float, default=0.2, help='策略噪声')
    parser.add_argument('--td3_policy_freq', type=int, default=2, help='策略更新频率')
    parser.add_argument('--td3_tau', type=float, default=0.005, help='软更新参数')
    parser.add_argument('--td3_gamma', type=float, default=0.99, help='TD3折扣因子')
    
    # 经验回放相关参数
    parser.add_argument('--use_per', type=bool, default=False, help='是否使用优先经验回放')  # 默认开启PER
    parser.add_argument('--use_her', type=bool, default=False, help='是否使用HER')  # 默认开启HER
    parser.add_argument('--buffer_size', type=int, default=1000000, help='回放缓冲区大小')
    parser.add_argument('--k_goals', type=int, default=4, help='HER中使用的额外目标数')
    parser.add_argument('--per_alpha', type=float, default=0.6, help='PER alpha参数')
    parser.add_argument('--per_beta', type=float, default=0.4, help='PER beta参数')
    parser.add_argument('--per_beta_increment', type=float, default=0.001, help='PER beta增长率')
    parser.add_argument('--per_epsilon', type=float, default=0.01, help='PER epsilon参数')
    parser.add_argument('--reward_scale', type=float, default=1.0, help='奖励缩放因子')
    
    args = parser.parse_args()

    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    # 测试代码的参数
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'td3'], default='td3',
                      help='选择测试算法(ppo或td3)')
    parser.add_argument('--env_type', type=str, choices=['env1', 'env2', 'env3', 'env4', 'env5'],
                      default='env5', help='选择环境类型')
    parser.add_argument('--render', type=bool, default=True,
                      help='是否渲染环境')
    parser.add_argument('--test_episodes', type=int, default=10,
                      help='测试回合数')
    
    # 模型加载参数
    parser.add_argument('--episode', type=int, default=400,
                      help='加载第几回合的模型，如果不指定则加载final模型')
    parser.add_argument('--model_dir', type=str, default='models/td3_env3',
                      help='模型所在目录')
    
    # PPO特定参数 (与训练时相同的默认值)
    parser.add_argument('--ppo_gamma', type=float, default=0.99, help='PPO折扣因子')
    parser.add_argument('--ppo_lam', type=float, default=0.95, help='GAE lambda参数')
    parser.add_argument('--ppo_clip_ratio', type=float, default=0.1, help='PPO剪切比例')
    parser.add_argument('--ppo_lr', type=float, default=3e-4, help='PPO学习率')
    parser.add_argument('--ppo_train_iters', type=int, default=80, help='每次更新的训练迭代次数')
    parser.add_argument('--ppo_target_kl', type=float, default=0.05, help='目标KL散度')
    parser.add_argument('--ppo_batch_size', type=int, default=5000, help='PPO batch大小')
    parser.add_argument('--ppo_minibatch_size', type=int, default=64, help='PPO minibatch大小')
    parser.add_argument('--ppo_max_ep_len', type=int, default=200, help='PPO最大回合长度')
    
    args = parser.parse_args()

    # 设置模型路径

    print("\n测试配置:")
    print(f"{'='*50}")
    print(f"算法: {args.algorithm}")
    print(f"环境: {args.env_type}")
    print(f"渲染: {'开启' if args.render else '关闭'}")
    print(f"测试回合数: {args.test_episodes}")
    if args.algorithm == 'ppo':
        print(f"PPO参数:")
        print(f"  折扣因子: {args.ppo_gamma}")
        print(f"  GAE lambda: {args.ppo_lam}")
        print(f"  剪切比例: {args.ppo_clip_ratio}")
        print(f"  最大回合长度: {args.ppo_max_ep_len}")
    print(f"{'='*50}\n")
    
    return args