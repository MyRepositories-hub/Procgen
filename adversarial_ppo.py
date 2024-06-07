import argparse
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from procgen import ProcgenEnv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules import build_encoder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip('.py'))
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--torch_deterministic', type=bool, default=True)
    parser.add_argument('--total_time_steps', type=int, default=int(5e7))
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--anneal_lr', type=bool, default=False)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--num_eval_workers', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=256)
    parser.add_argument('--num_mini_batches', type=int, default=8)
    parser.add_argument('--update_epochs', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--norm_adv', type=bool, default=True)
    parser.add_argument('--clip_value_loss', type=bool, default=True)
    parser.add_argument('--c_1', type=float, default=0.5)
    parser.add_argument('--c_2', type=float, default=0.01)
    parser.add_argument('--c_3', type=float, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_mini_batches)
    args.num_iterations = int(args.total_time_steps // args.batch_size)
    return args


def make_env(envs, gamma):
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs['rgb'])
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    return envs


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = build_encoder(latent_dim)

    def forward(self, x):
        return self.encoder(x.permute((0, 3, 1, 2)) / 255.0)  # [B, H, W, C] -> [B, C, H, W]


class Agent(nn.Module):
    def __init__(self, envs, latent_dim):
        super().__init__()
        self.single_observation_shape = envs.single_observation_space.shape
        self.actor = layer_init(nn.Linear(latent_dim, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(latent_dim, 1), std=1)

    def get_value(self, encoder, x):
        return self.critic(encoder(x))

    def get_action_and_value(self, encoder, x, a=None, freeze=False):
        hidden = encoder(x)
        if freeze:
            with torch.no_grad():
                actor_output = self.actor(hidden)
                distribution = Categorical(logits=actor_output)
                if a is None:
                    a = distribution.sample()
        else:
            actor_output = self.actor(hidden)
            distribution = Categorical(logits=actor_output)
            if a is None:
                a = distribution.sample()
        return a, distribution.log_prob(a), distribution.entropy(), self.critic(hidden), distribution.probs


def compute_kld(p, q):
    return torch.sum(p * (p.log() - q.log()), -1)


def main(env_id, seed):
    args = get_args()
    args.env_id = env_id
    args.seed = seed

    run_name_1 = (
        'ppo_' + str(args.c_3) + '_agent_1' +
        '_epoch_' + str(args.update_epochs) +
        '_seed_' + str(args.seed)
    )
    run_name_2 = (
        'ppo_' + str(args.c_3) + '_agent_2' +
        '_epoch_' + str(args.update_epochs) +
        '_seed_' + str(args.seed)
    )

    # Save training logs
    path_string_1 = str(args.env_id) + '/' + run_name_1
    writer_1 = SummaryWriter(path_string_1)
    eval_writer_1 = SummaryWriter(path_string_1 + '_eval')
    writer_1.add_text(
        'Hyperparameter',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()])),
    )
    path_string_2 = str(args.env_id) + '/' + run_name_2
    writer_2 = SummaryWriter(path_string_2)
    eval_writer_2 = SummaryWriter(path_string_2 + '_eval')
    writer_2.add_text(
        'Hyperparameter',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()])),
    )

    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Initialize environments
    envs_1 = ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_id,
        rand_seed=args.seed,
        num_levels=500,
        start_level=0,
        distribution_mode='hard'
    )
    envs_1 = make_env(envs_1, args.gamma)
    envs_1.single_action_space = envs_1.action_space
    envs_1.single_observation_space = envs_1.observation_space['rgb']
    envs_2 = ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_id,
        rand_seed=args.seed,
        num_levels=500,
        start_level=0,
        distribution_mode='hard'
    )
    envs_2 = make_env(envs_2, args.gamma)
    envs_2.single_action_space = envs_2.action_space
    envs_2.single_observation_space = envs_2.observation_space['rgb']

    # Initialize encoders
    encoder_1 = Encoder(256).to(device)
    encoder_2 = Encoder(256).to(device)

    # Initialize agents
    agent_1 = Agent(envs_1, 256).to(device)
    parameters_1 = list(agent_1.parameters()) + list(encoder_1.parameters()) + list(encoder_2.parameters())
    optimizer_1 = optim.Adam(parameters_1, lr=args.learning_rate, eps=1e-5)
    agent_2 = Agent(envs_2, 256).to(device)
    parameters_2 = list(agent_2.parameters()) + list(encoder_1.parameters()) + list(encoder_2.parameters())
    optimizer_2 = optim.Adam(parameters_2, lr=args.learning_rate, eps=1e-5)

    # Initialize buffer
    obs_1 = torch.zeros((args.num_steps, args.num_envs) + envs_1.single_observation_space.shape).to(device)
    actions_1 = torch.zeros((args.num_steps, args.num_envs)).to(device)
    log_probs_1 = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_1 = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_1 = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_1 = torch.zeros((args.num_steps, args.num_envs)).to(device)
    obs_2 = torch.zeros((args.num_steps, args.num_envs) + envs_2.single_observation_space.shape).to(device)
    actions_2 = torch.zeros((args.num_steps, args.num_envs)).to(device)
    log_probs_2 = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_2 = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_2 = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_2 = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Data collection
    global_step = 0
    start_time = time.time()

    next_obs_1 = torch.Tensor(envs_1.reset()).to(device)
    next_done_1 = torch.zeros(args.num_envs).to(device)
    next_obs_2 = torch.Tensor(envs_2.reset()).to(device)
    next_done_2 = torch.zeros(args.num_envs).to(device)

    for iteration in tqdm(range(1, args.num_iterations + 1)):

        # Linear decay of learning rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lr_now = frac * args.learning_rate
            optimizer_1.param_groups[0]['lr'] = lr_now
            optimizer_2.param_groups[0]['lr'] = lr_now

        for step in range(0, args.num_steps):
            global_step += args.num_envs

            obs_1[step] = next_obs_1
            dones_1[step] = next_done_1
            obs_2[step] = next_obs_2
            dones_2[step] = next_done_2

            # Compute the logarithm of the action probability output by the old policy network
            with torch.no_grad():
                action_1, log_prob_1, _, value_1, prob_1 = agent_1.get_action_and_value(encoder_1, next_obs_1)
                values_1[step] = value_1.flatten()
                action_2, log_prob_2, _, value_2, prob_2 = agent_2.get_action_and_value(encoder_2, next_obs_2)
                values_2[step] = value_2.flatten()

            actions_1[step] = action_1
            log_probs_1[step] = log_prob_1
            actions_2[step] = action_2
            log_probs_2[step] = log_prob_2

            # Update the environments
            next_obs_1, reward_1, next_done_1, info_1 = envs_1.step(action_1.cpu().numpy())
            rewards_1[step] = torch.tensor(reward_1).to(device).view(-1)
            next_obs_1, next_done_1 = torch.Tensor(next_obs_1).to(device), torch.Tensor(next_done_1).to(device)
            next_obs_2, reward_2, next_done_2, info_2 = envs_2.step(action_2.cpu().numpy())
            rewards_2[step] = torch.tensor(reward_2).to(device).view(-1)
            next_obs_2, next_done_2 = torch.Tensor(next_obs_2).to(device), torch.Tensor(next_done_2).to(device)

            for item in info_1:
                if 'episode' in item.keys():
                    writer_1.add_scalar('charts/episodic_return', item['episode']['r'], global_step)
                    break
            for item in info_2:
                if 'episode' in item.keys():
                    writer_2.add_scalar('charts/episodic_return', item['episode']['r'], global_step)
                    break

        # Use GAE (Generalized Advantage Estimation) technique to estimate the advantage function
        with torch.no_grad():
            next_value_1 = agent_1.get_value(encoder_1, next_obs_1).reshape(1, -1)
            advantages_1 = torch.zeros_like(rewards_1).to(device)
            last_gae_lam_1 = 0
            next_value_2 = agent_2.get_value(encoder_2, next_obs_2).reshape(1, -1)
            advantages_2 = torch.zeros_like(rewards_2).to(device)
            last_gae_lam_2 = 0
            for t in reversed(range(args.num_steps)):
                next_non_terminal_1 = 1.0 - next_done_1 if t == args.num_steps - 1 else 1.0 - dones_1[t + 1]
                next_values_1 = next_value_1 if t == args.num_steps - 1 else values_1[t + 1]
                delta_1 = rewards_1[t] + args.gamma * next_values_1 * next_non_terminal_1 - values_1[t]
                advantages_1[t] = last_gae_lam_1 = (
                    delta_1 + args.gamma * args.gae_lambda * next_non_terminal_1 * last_gae_lam_1
                )
                next_non_terminal_2 = 1.0 - next_done_2 if t == args.num_steps - 1 else 1.0 - dones_2[t + 1]
                next_values_2 = next_value_2 if t == args.num_steps - 1 else values_2[t + 1]
                delta_2 = rewards_2[t] + args.gamma * next_values_2 * next_non_terminal_2 - values_2[t]
                advantages_2[t] = last_gae_lam_2 = (
                    delta_2 + args.gamma * args.gae_lambda * next_non_terminal_2 * last_gae_lam_2
                )
            returns_1 = advantages_1 + values_1
            returns_2 = advantages_2 + values_2

        # ---------------------- We have collected enough data, now let's start training ---------------------- #
        # Flatten each batch
        b_obs_1 = obs_1.reshape((-1,) + envs_1.single_observation_space.shape)
        b_actions_1 = actions_1.reshape(-1)
        b_log_probs_1 = log_probs_1.reshape(-1)
        b_advantages_1 = advantages_1.reshape(-1)
        b_returns_1 = returns_1.reshape(-1)
        b_values_1 = values_1.reshape(-1)
        b_obs_2 = obs_2.reshape((-1,) + envs_2.single_observation_space.shape)
        b_actions_2 = actions_2.reshape(-1)
        b_log_probs_2 = log_probs_2.reshape(-1)
        b_advantages_2 = advantages_2.reshape(-1)
        b_returns_2 = returns_2.reshape(-1)
        b_values_2 = values_2.reshape(-1)

        # Update the policy network and value network
        b_index = np.arange(args.batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(b_index)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_index = b_index[start:end]

                # Here i_j_k means s_i -> phi_j -> pi_k
                # Agent 1
                _, new_log_prob_1_1_1, entropy_1_1_1, new_value_1_1_1, new_prob_1_1_1 = agent_1.get_action_and_value(
                    encoder_1, b_obs_1[mb_index], b_actions_1.long()[mb_index]
                )
                _, _, _, _, new_prob_1_2_1 = agent_1.get_action_and_value(
                    encoder_2, b_obs_1[mb_index]
                )
                _, _, _, _, new_prob_1_1_2 = agent_2.get_action_and_value(
                    encoder_1, b_obs_1[mb_index], freeze=True
                )
                _, _, _, _, new_prob_1_2_2 = agent_2.get_action_and_value(
                    encoder_2, b_obs_1[mb_index], freeze=True
                )
                # Agent 2
                _, new_log_prob_2_2_2, entropy_2_2_2, new_value_2_2_2, new_prob_2_2_2 = agent_2.get_action_and_value(
                    encoder_2, b_obs_2[mb_index], b_actions_2.long()[mb_index]
                )
                _, _, _, _, new_prob_2_1_2 = agent_2.get_action_and_value(
                    encoder_1, b_obs_2[mb_index]
                )
                _, _, _, _, new_prob_2_2_1 = agent_1.get_action_and_value(
                    encoder_2, b_obs_2[mb_index], freeze=True
                )
                _, _, _, _, new_prob_2_1_1 = agent_1.get_action_and_value(
                    encoder_1, b_obs_2[mb_index], freeze=True
                )

                # Compute kld
                # Agent 1
                kld_own_1 = compute_kld(new_prob_1_1_1, new_prob_1_2_1).mean()
                kld_other_1 = compute_kld(new_prob_1_2_2, new_prob_1_1_2).mean()
                total_kld_loss_1 = kld_own_1 - kld_other_1
                # Agent 2
                kld_own_2 = compute_kld(new_prob_2_2_2, new_prob_2_1_2).mean()
                kld_other_2 = compute_kld(new_prob_2_1_1, new_prob_2_2_1).mean()
                total_kld_loss_2 = kld_own_2 - kld_other_2

                # Ratio
                log_ratio_1 = new_log_prob_1_1_1 - b_log_probs_1[mb_index]
                ratios_1 = log_ratio_1.exp()
                log_ratio_2 = new_log_prob_2_2_2 - b_log_probs_2[mb_index]
                ratios_2 = log_ratio_2.exp()

                # Advantage normalization
                mb_advantages_1 = b_advantages_1[mb_index]
                mb_advantages_2 = b_advantages_2[mb_index]
                if args.norm_adv:
                    mb_advantages_1 = (mb_advantages_1 - mb_advantages_1.mean()) / (mb_advantages_1.std() + 1e-12)
                    mb_advantages_2 = (mb_advantages_2 - mb_advantages_2.mean()) / (mb_advantages_2.std() + 1e-12)

                # Policy loss
                # Agent 1
                policy_loss_1_1 = -mb_advantages_1 * ratios_1
                policy_loss_1_2 = -mb_advantages_1 * torch.clamp(ratios_1, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
                policy_loss_1 = torch.max(policy_loss_1_1, policy_loss_1_2).mean()
                # Agent 2
                policy_loss_2_1 = -mb_advantages_2 * ratios_2
                policy_loss_2_2 = -mb_advantages_2 * torch.clamp(ratios_2, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
                policy_loss_2 = torch.max(policy_loss_2_1, policy_loss_2_2).mean()

                # Value loss
                # Agent 1
                new_value_1_1_1 = new_value_1_1_1.view(-1)
                if args.clip_value_loss:
                    value_loss_un_clipped_1 = (new_value_1_1_1 - b_returns_1[mb_index]) ** 2
                    value_clipped_1 = b_values_1[mb_index] + torch.clamp(
                        new_value_1_1_1 - b_values_1[mb_index],
                        -args.clip_epsilon,
                        args.clip_epsilon
                    )
                    value_loss_clipped_1 = (value_clipped_1 - b_returns_1[mb_index]) ** 2
                    value_loss_max_1 = torch.max(value_loss_un_clipped_1, value_loss_clipped_1)
                    value_loss_1 = 0.5 * value_loss_max_1.mean()
                else:
                    value_loss_1 = 0.5 * ((new_value_1_1_1 - b_returns_1[mb_index]) ** 2).mean()
                # Agent 2
                new_value_2_2_2 = new_value_2_2_2.view(-1)
                if args.clip_value_loss:
                    value_loss_un_clipped_2 = (new_value_2_2_2 - b_returns_2[mb_index]) ** 2
                    value_clipped_2 = b_values_2[mb_index] + torch.clamp(
                        new_value_2_2_2 - b_values_2[mb_index],
                        -args.clip_epsilon,
                        args.clip_epsilon
                    )
                    value_loss_clipped_2 = (value_clipped_2 - b_returns_2[mb_index]) ** 2
                    value_loss_max_2 = torch.max(value_loss_un_clipped_2, value_loss_clipped_2)
                    value_loss_2 = 0.5 * value_loss_max_2.mean()
                else:
                    value_loss_2 = 0.5 * ((new_value_2_2_2 - b_returns_2[mb_index]) ** 2).mean()

                # Policy entropy
                entropy_loss_1 = entropy_1_1_1.mean()
                entropy_loss_2 = entropy_2_2_2.mean()

                # Save the data during the training process
                writer_1.add_scalar('charts/kld_own', kld_own_1.item(), global_step)
                writer_1.add_scalar('charts/kld_other', kld_other_1.item(), global_step)
                writer_1.add_scalar('losses/policy_loss', policy_loss_1.item(), global_step)
                writer_1.add_scalar('losses/value_loss', value_loss_1.item(), global_step)
                writer_1.add_scalar('losses/entropy', entropy_loss_1.item(), global_step)
                writer_1.add_scalar('losses/total_kld_loss', total_kld_loss_1.item(), global_step)

                writer_2.add_scalar('charts/kld_own', kld_own_2.item(), global_step)
                writer_2.add_scalar('charts/kld_other', kld_other_2.item(), global_step)
                writer_2.add_scalar('losses/policy_loss', policy_loss_2.item(), global_step)
                writer_2.add_scalar('losses/value_loss', value_loss_2.item(), global_step)
                writer_2.add_scalar('losses/entropy', entropy_loss_2.item(), global_step)
                writer_2.add_scalar('losses/total_kld_loss', total_kld_loss_2.item(), global_step)

                def compute_total_loss(policy_loss, value_loss, entropy_loss, total_kld_loss):
                    return policy_loss + value_loss * args.c_1 - entropy_loss * args.c_2 + total_kld_loss * args.c_3

                # Update network parameters
                total_loss_1 = compute_total_loss(policy_loss_1, value_loss_1, entropy_loss_1, total_kld_loss_1)
                optimizer_1.zero_grad()
                total_loss_1.backward()
                nn.utils.clip_grad_norm_(parameters_1, args.max_grad_norm)
                optimizer_1.step()

                total_loss_2 = compute_total_loss(policy_loss_2, value_loss_2, entropy_loss_2, total_kld_loss_2)
                optimizer_2.zero_grad()
                total_loss_2.backward()
                nn.utils.clip_grad_norm_(parameters_2, args.max_grad_norm)
                optimizer_2.step()

        # Save the data during the training process
        writer_1.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)
        writer_2.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)

        # Evaluation
        eval_envs_1 = ProcgenEnv(
            num_envs=args.num_eval_workers,
            env_name=args.env_id,
            rand_seed=1024,
            num_levels=0,
            start_level=0,
            distribution_mode='hard'
        )
        eval_envs_1 = make_env(eval_envs_1, args.gamma)
        eval_next_obs_1 = torch.Tensor(eval_envs_1.reset()).to(device)
        eval_episodic_returns_1 = []
        while len(eval_episodic_returns_1) < 10:
            with torch.no_grad():
                eval_action_1, _, _, _, _ = agent_1.get_action_and_value(encoder_1, eval_next_obs_1)
            eval_next_obs_1, _, _, eval_info_1 = eval_envs_1.step(eval_action_1.cpu().numpy())
            eval_next_obs_1 = torch.Tensor(eval_next_obs_1).to(device)
            for item in eval_info_1:
                if 'episode' in item.keys():
                    eval_episodic_returns_1.append(item['episode']['r'])
        eval_writer_1.add_scalar('charts/episodic_return', np.mean(eval_episodic_returns_1), global_step)
        eval_envs_1.close()

        eval_envs_2 = ProcgenEnv(
            num_envs=args.num_eval_workers,
            env_name=args.env_id,
            rand_seed=1024,
            num_levels=0,
            start_level=0,
            distribution_mode='hard'
        )
        eval_envs_2 = make_env(eval_envs_2, args.gamma)
        eval_next_obs_2 = torch.Tensor(eval_envs_2.reset()).to(device)
        eval_episodic_returns_2 = []
        while len(eval_episodic_returns_2) < 10:
            with torch.no_grad():
                eval_action_2, _, _, _, _ = agent_2.get_action_and_value(encoder_2, eval_next_obs_2)
            eval_next_obs_2, _, _, eval_info_2 = eval_envs_2.step(eval_action_2.cpu().numpy())
            eval_next_obs_2 = torch.Tensor(eval_next_obs_2).to(device)
            for item in eval_info_2:
                if 'episode' in item.keys():
                    eval_episodic_returns_2.append(item['episode']['r'])
        eval_writer_2.add_scalar('charts/episodic_return', np.mean(eval_episodic_returns_2), global_step)
        eval_envs_2.close()

    envs_1.close()
    envs_2.close()
    writer_1.close()
    writer_2.close()


def run():
    for env_id in [
        # 'bigfish',
        # 'bossfight',
        # 'caveflyer',
        # 'chaser',
        # 'climber',
        # 'coinrun',
        'dodgeball',
        'fruitbot',
        'heist',
        'jumper',
        'leaper',
        'maze',
        'miner',
        'ninja',
        'plunder',
        'starpilot'
    ]:
        for seed in [1]:
            print(env_id, 'seed:', seed)
            main(env_id, seed)


if __name__ == '__main__':
    run()
