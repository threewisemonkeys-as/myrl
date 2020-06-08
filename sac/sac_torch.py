# Soft Actor Critic in pytorch
# Atharv Sonwane <atharvs.twm@gmail.com>

# References -
# https://arxiv.org/pdf/1801.01290.pdf
# https://spinningup.openai.com/en/latest/algorithms/sac.html


import torch
import torch.nn as nn
import numpy as np
from collections import deque, namedtuple
from itertools import count
import matplotlib.pyplot as plt
import time
import copy
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Hyperparameters
TIMESTEPS = 5000
Q_HIDDEN_DIM = 32
POLICY_HIDDEN_DIM = 32
Q_LEARNING_RATE = 1e-3
P_LEARNING_RATE = 1e-3
GAMMA = 0.99
ENTROPY_TEMP = 0.02 # Entropy regularizer. Equivalent to inverse of reward scale
BATCH_SIZE = 128
POLYAK_CONST = 0.995
MAX_BUFFER_SIZE = 3000
STEPS_TO_UPDATE_AFTER = TIMESTEPS / 10
UPDATE_INTERVAL = 1
STEPS_TO_SELECT_FROM_MODEL_AFTER = TIMESTEPS / 5
MAX_TRAJ_LENGTH = 1000  # For pendulum this is 200
SAVE_FREQUENCY = None
EPS = 1e-4
LOG_MAX = 2
LOG_MIN = -20

# Transition tuple
Transition = namedtuple(
    "Transition", ("observation", "action", "reward", "next_observation", "done")
)


class Actor(nn.Module):
    def __init__(self, s_dim, hidden_dim, a_dim, upper_lim, lower_lim):
        super(Actor, self).__init__()
        self.scale_factor = upper_lim - lower_lim
        self.range_midpoint = (upper_lim + lower_lim) / 2
        self.common_model = nn.Sequential(
                nn.Linear(s_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ).to(device).to(dtype)
        self.mean_head = nn.Linear(hidden_dim, a_dim).to(device).to(dtype)
        self.std_dev_head = nn.Linear(hidden_dim, a_dim).to(device).to(dtype)

    def forward(self, x):
        x = self.common_model(x)
        mean = self.mean_head(x)
        log_std_dev = self.std_dev_head(x)
        log_std_dev = torch.clamp(log_std_dev, min=LOG_MIN, max=LOG_MAX)
        return mean, log_std_dev

    def sample_action(self, s, get_logprob=True, deterministic=False):
        # compute mean and log of std dev from forward pass of network
        mean, log_std_dev = self.forward(s)
        std_dev = torch.exp(log_std_dev)
        action_dist = torch.distributions.Normal(mean, std_dev)

        # return mean for testing
        if deterministic:
            action = mean
        else:
            action = action_dist.rsample()        

        # compute log probabilities if required and appply correction for tanh squashing
        if get_logprob:
            logprob = action_dist.log_prob(action)

            # Appendix C of https://arxiv.org/pdf/1801.01290.pdf for correction
            # logprob -= torch.log(1 - action.pow(2) + EPS)

            # More numerically stable version of the correction - 
            # https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac/core.py#L60
            logprob -= (2*(np.log(2) - logprob - nn.functional.softplus(-2*logprob)))

            logprob = logprob.sum(axis=-1, keepdim=True)

        else:
            logprob = None
    
        # squash and rescale action value
        action = torch.tanh(action) * self.scale_factor + self.range_midpoint

        return action, logprob


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dim):
        super(Critic, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.model = nn.Sequential(
                nn.Linear(s_dim + a_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            ).to(device).to(dtype)

    def forward(self, s, a):
        x = torch.cat([s.view(-1, self.s_dim), a.view(-1, self.a_dim)], dim=1)
        return self.model(x)


# Model
class SAC:
    def __init__(
        self, env, q_hidden_dim=Q_HIDDEN_DIM, policy_hidden_dim=POLICY_HIDDEN_DIM,
    ):

        self.env = env
        self.action_shape = env.action_space.shape
        self.observation_shape = env.observation_space.shape
        self.action_min = env.action_space.low[0]
        self.action_max = env.action_space.high[0]

        Q1 = Critic(self.observation_shape[0], self.action_shape[0], q_hidden_dim)
        Q2 = Critic(self.observation_shape[0], self.action_shape[0], q_hidden_dim)
        target_Q1 = copy.deepcopy(Q1)
        target_Q2 = copy.deepcopy(Q2)
        self.Qs = [Q1, Q2]
        self.target_Qs = [target_Q1, target_Q2]
        for target_Q in self.target_Qs:
            for p in target_Q.parameters():
                p.requires_grad = False

        self.policy = Actor(
            self.observation_shape[0], policy_hidden_dim, self.action_shape[0], self.action_max, self.action_min
        )

    def _select_action(self, observation, select_after=0, step_count=0, deterministic=False):
        if step_count < select_after:
            action = torch.tensor(env.action_space.sample(), device=device, dtype=dtype).unsqueeze(0)
        else:
            with torch.no_grad():
                action, _ = self.policy.sample_action(observation, get_logprob=False, deterministic=deterministic)
                # import pdb; pdb.set_trace();


        return action

    def _update(
        self,
        replay_buffer,
        batch_size,
        q_optimizers,
        policy_optimizer,
        timestep,
        gamma,
        polyak_const,
        temp
    ):

        # Get a batch of samples and unwrap them
        sample = Transition(
            *[torch.cat(i) for i in [*zip(*random.sample(replay_buffer, batch_size))]]
        )

        # Compute the target Q values for each state action pair in batch
        with torch.no_grad():
            # Compute nexy action by sample from policy
            next_action, logprob = self.policy.sample_action(sample.next_observation)
            # Use minimum next state Q value estimate for computing target
            next_state_q_val = torch.min(*[Q(sample.next_observation, next_action) for Q in self.target_Qs])
            # Compute targets using entropy objective
            target_q_vals = sample.reward + gamma * (next_state_q_val - temp * logprob) * (~sample.done)

        # import pdb; pdb.set_trace();
        
        # Update all the Q networks
        q_loss = 0.0
        for Q, target_Q, q_optimizer in zip(self.Qs, self.target_Qs, q_optimizers):
           
            # Compute the current Q values for each state action pair in batch
            q_vals = Q(sample.observation, sample.action)

            # Compute the q loss, backpropogate and update the gradients
            q_optimizer.zero_grad()
            curr_q_loss = nn.MSELoss()(target_q_vals, q_vals)
            curr_q_loss.backward()
            q_optimizer.step()
            q_loss += float(curr_q_loss)

        # Turn off gradient calculation for Q networks to save compute during policy update
        for Q in self.Qs:
            for p in Q.parameters():
                p.requires_grad = False

        # Compute the policy loss, backpropogate and update the gradients
        policy_optimizer.zero_grad()
        action, logprob = self.policy.sample_action(sample.observation)
        min_q_val =  torch.min(*[Q(sample.observation, action) for Q in self.Qs])
        policy_loss = -1 * torch.mean(min_q_val - temp * logprob)
        policy_loss.backward()
        policy_optimizer.step()

        # Turn on gradient calculation for Q networks after policy update
        for Q in self.Qs:
            for p in Q.parameters():
                p.requires_grad = True

        # Update target q networks with polyak averaging
        for Q, target_Q in zip(self.Qs, self.target_Qs):
            with torch.no_grad():
                for p_target, p in zip(target_Q.parameters(), Q.parameters()):
                    p_target.data.mul_(polyak_const)
                    p_target.data.add_((1 - polyak_const) * p.data)
                    
        return q_loss, policy_loss.item()

    def train(
        self,
        timesteps=TIMESTEPS,
        q_lr=Q_LEARNING_RATE,
        p_lr=P_LEARNING_RATE,
        gamma=GAMMA,
        temp=ENTROPY_TEMP,
        batch_size=BATCH_SIZE,
        polyak_const=POLYAK_CONST,
        max_buffer_size=MAX_BUFFER_SIZE,
        update_after=STEPS_TO_UPDATE_AFTER,
        update_every=UPDATE_INTERVAL,
        select_after=STEPS_TO_SELECT_FROM_MODEL_AFTER,
        max_traj_length=MAX_TRAJ_LENGTH,
        save_freq=SAVE_FREQUENCY,
        render=False,
        plot_rewards=True,
        VERBOSE=False,
        PLOT_REWARDS=False,
    ):
        """ Trains q and policy network """

        print(
            f"\nTraining model for {timesteps} timesteps with - \n"
            f"q learning rate: {q_lr}\n"
            f"policy learning rate: {p_lr}\n"
            f"gamma:  {gamma}\n"
            f"batch size:  {batch_size}\n"
            f"polyak constant:  {polyak_const}\n"
            f"maximum buffer capacity: {max_buffer_size}\n"
            f"minimum steps to select random for: {select_after}\n"
            f"minimum steps before updating network:  {update_after}\n"
            f"interval for updating: {update_every}\n"
            f"maximum trajectory length:  {max_traj_length}\n"
        )
        start_time = time.time()
        self.policy.train()
        for Q in self.Qs:
            Q.train()
        q_optimizers = [torch.optim.Adam(Q.parameters(), lr=q_lr) for Q in self.Qs]
        policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=p_lr)
        replay_buffer = deque(maxlen=max_buffer_size)
        rewards = []
        step_count = 0

        for episode in count():

            observation = self.env.reset()
            observation = torch.tensor(
                observation, device=device, dtype=dtype
            ).unsqueeze(0)
            done = torch.tensor([False], device=device, dtype=torch.bool).unsqueeze(0)
            episode_rewards = []

            for _ in range(max_traj_length):
                step_count += 1
                if render:
                    self.env.render()

                action = self._select_action(observation, select_after, step_count, deterministic=False)
                next_observation, reward, done, _ = self.env.step(action[0])
                episode_rewards.append(float(reward))
                next_observation = torch.tensor(
                    next_observation, device=device, dtype=dtype
                ).unsqueeze(0)
                reward = torch.tensor([reward], device=device, dtype=dtype).unsqueeze(0)
                done = torch.tensor([done], device=device, dtype=torch.bool).unsqueeze(0)

                transition = Transition(
                    observation, action, reward, next_observation, done
                )
                replay_buffer.append(transition)
                observation = next_observation

                # Update the Deep Q Network if sufficient transitions available every interval
                if step_count >= update_after and step_count % update_every == 0:
                    q_loss, policy_loss = self._update(
                        replay_buffer,
                        batch_size,
                        q_optimizers,
                        policy_optimizer,
                        timesteps,
                        gamma,
                        polyak_const,
                        temp
                    )

                if save_freq is not None:
                    if step_count >= update_after and step_count % save_freq == 0:
                        model.save(f"models/models/ddpg_torch_{step_count}")

                if done or step_count == timesteps:
                    break

            if step_count == timesteps:
                break

            # Log rewards and losses
            total_episode_reward = sum(episode_rewards)
            rewards.append(total_episode_reward)
            if VERBOSE:
                print(
                    f"Episode {episode+1}: Step Count = {step_count} | Reward = {total_episode_reward:.2f} | ",
                    end="",
                )
                if step_count >= update_after:
                    print(f" Q Loss = {q_loss:.2f} | Policy Loss = {policy_loss:.2f}")
                else:
                    print("Collecting Experience")
            
        print(f"\nTraining Completed in {(time.time() - start_time):.2f} seconds")
        if PLOT_REWARDS:
            plt.plot(rewards)
            plt.savefig("sac_reward_plot.png")
        env.close()

    def save(self, path):
        """ Save model parameters """

        torch.save(
            {
                "q1_state_dict": self.Qs[0].state_dict(),
                "q2_state_dict": self.Qs[1].state_dict(),
                "policy_state_dict": self.policy.state_dict(),
            },
            path,
        )
        print(f"\nSaved model parameters to {path}")

    def load(self, path):
        """ Load model parameters """

        checkpoint = torch.load(path)
        self.Q[0].load_state_dict(checkpoint["q1_state_dict"])
        self.Q[1].load_state_dict(checkpoint["q2_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"\nLoaded model parameters from {path}")

    def eval(self, episodes, deterministic=True, render=False):
        """ Evaluates model performance """

        print(f"\nEvaluating model for {episodes} episodes ...\n")
        start_time = time.time()
        self.policy.eval()
        rewards = []

        for episode in range(episodes):

            observation = self.env.reset()
            observation = torch.tensor(observation, device=device, dtype=dtype)
            done = False
            episode_rewards = []

            while not done:
                if render:
                    self.env.render()

                action = self._select_action(observation, deterministic=deterministic)
                next_observation, reward, done, _ = self.env.step(action.detach())
                episode_rewards.append(reward)
                next_observation = torch.tensor(
                    next_observation, device=device, dtype=dtype
                )
                observation = next_observation

            total_episode_reward = sum(episode_rewards)
            rewards.append(total_episode_reward)
            print(
                f"Episode {episode+1}: Total Episode Reward = {total_episode_reward:.2f}"
            )
            rewards.append(total_episode_reward)

        env.close()
        print(f"\nAverage Reward for an episode = {np.mean(rewards):.2f}")
        print(f"Evaluation Completed in {(time.time() - start_time):.2f} seconds")


if __name__ == "__main__":
    import gym
    import random

    from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv

    # Load environment
    env_name = "RacecarBulletEnv"; env = RacecarGymEnv(renders=False, isDiscrete=False)
    # env_name = "Pendulum-v0"; env = gym.make(env_name)
    # env_name = "MountainCarContinuous-v0"; env = gym.make(env_name)

    print(
        f"Env: {env_name} |"
        f" Observation Space: {env.observation_space} |"
        f" Action Space: {env.action_space}"
    )

    model = SAC(env)
    model.train(VERBOSE=True, PLOT_REWARDS=True)
    model.save(f"models/sac_torch_{env_name}.pt")
    # model.load(f"models/sac_torch_{env_name}.pt")
    model.eval(10, render=True)
