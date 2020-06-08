# Deep Deterministic Policy Gradient in pytorch
# Atharv Sonwane <atharvs.twm@gmail.com>

# References -
# http://proceedings.mlr.press/v32/silver14.pdf
# https://arxiv.org/pdf/1509.02971.pdf
# https://spinningup.openai.com/en/latest/algorithms/ddpg.html

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
TIMESTEPS = 10000
Q_HIDDEN_DIM = 32
POLICY_HIDDEN_DIM = 32
Q_LEARNING_RATE = 1e-3
P_LEARNING_RATE = 1e-3
GAMMA = 0.99
BATCH_SIZE = 128
POLYAK_CONST = 0.995
NOISE_STD_DEV = 0.2
MAX_BUFFER_SIZE = 5000
STEPS_TO_UPDATE_AFTER = 1000
UPDATE_INTERVAL = 1
STEPS_TO_SELECT_FROM_MODEL_AFTER = 2000
MAX_TRAJ_LENGTH = 1000  # For pendulum this is 200
SAVE_FREQUENCY = None

# Transition tuple
Transition = namedtuple(
    "Transition", ("observation", "action", "reward", "next_observation", "done")
)


class Actor(nn.Module):
    def __init__(self, s_dim, hidden_dim, a_dim, lim):
        super(Actor, self).__init__()
        self.lim = lim
        self.model = nn.Sequential(
                nn.Linear(s_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, a_dim),
                nn.Tanh(),
            ).to(device).to(dtype)

    def forward(self, a):
        return self.model(a) * self.lim


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
class DDPG:
    def __init__(
        self, env, q_hidden_dim=Q_HIDDEN_DIM, policy_hidden_dim=POLICY_HIDDEN_DIM,
    ):

        self.env = env
        self.action_shape = env.action_space.shape
        self.observation_shape = env.observation_space.shape
        self.action_min = env.action_space.low[0]
        self.action_max = env.action_space.high[0]

        self.Q = Critic(self.observation_shape[0], self.action_shape[0], q_hidden_dim)
        self.target_Q = copy.deepcopy(self.Q)
        for p in self.target_Q.parameters():
            p.requires_grad = False

        self.policy = Actor(
            self.observation_shape[0], policy_hidden_dim, self.action_shape[0], self.action_max
        )
        self.target_policy = copy.deepcopy(self.policy)
        for p in self.target_policy.parameters():
            p.requires_grad = False

    def _select_action(self, observations, noise=0, select_after=0, step_count=0):
        if step_count < select_after:
            action = torch.tensor(env.action_space.sample(), device=device, dtype=dtype).unsqueeze(0)
        else:
            with torch.no_grad():
                noisy_action = self.policy(observations) + noise * torch.randn(size=self.action_shape)
                action = torch.clamp(noisy_action, self.action_min, self.action_max)

        return action

    def _update(
        self,
        replay_buffer,
        batch_size,
        q_optimizer,
        policy_optimizer,
        gamma,
        polyak_const,
    ):

        # Get a batch of samples and unwrap them
        sample = Transition(
            *[torch.cat(i) for i in [*zip(*random.sample(replay_buffer, batch_size))]]
        )

        # Compute the policy loss, backpropogate and update the gradients
        policy_optimizer.zero_grad()
        policy_loss = -1 * torch.mean(
            self.Q(sample.observation, self.policy(sample.observation))
        )
        policy_loss.backward()
        policy_optimizer.step()

        # Compute the target Q values for each state action pair in batch
        with torch.no_grad():
            target_q_vals = sample.reward + gamma * self.target_Q(
                sample.next_observation, self.target_policy(sample.next_observation)
            ) * (~sample.done)

        # import pdb; pdb.set_trace();
        

        # Compute the current Q values for each state action pair in batch
        q_vals = self.Q(sample.observation, sample.action)

        # Compute the q loss, backpropogate and update the gradients
        q_optimizer.zero_grad()
        q_loss = nn.MSELoss()(target_q_vals, q_vals)
        q_loss.backward()
        q_optimizer.step()


        # Update target q and policy networks with polyak averaging
        with torch.no_grad():
            for p_target, p in zip(self.target_Q.parameters(), self.Q.parameters()):
                p_target.data = polyak_const * p_target.data + (1 - polyak_const) * p.data
            for p_target, p in zip(
                self.target_policy.parameters(), self.policy.parameters()
            ):
                p_target.data = polyak_const * p_target.data + (1 - polyak_const) * p.data

        return q_loss.item(), policy_loss.item()

    def train(
        self,
        timesteps=TIMESTEPS,
        q_lr=Q_LEARNING_RATE,
        p_lr=P_LEARNING_RATE,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        polyak_const=POLYAK_CONST,
        noise=NOISE_STD_DEV,
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
            f"std. dev. of noise: {noise}"
            f"maximum buffer capacity: {max_buffer_size}\n"
            f"minimum steps to select random for: {select_after}\n"
            f"minimum steps before updating network:  {update_after}\n"
            f"interval for updating: {update_every}\n"
            f"maximum trajectory length:  {max_traj_length}\n"
        )
        start_time = time.time()
        self.Q.train()
        self.policy.train()
        q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=q_lr)
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

                action = self._select_action(observation, noise, select_after, step_count)
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
                        q_optimizer,
                        policy_optimizer,
                        gamma,
                        polyak_const,
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
            plt.savefig("ddpg_reward_plot.png")
        env.close()

    def save(self, path):
        """ Save model parameters """

        torch.save(
            {
                "q_state_dict": self.Q.state_dict(),
                "policy_state_dict": self.policy.state_dict(),
            },
            path,
        )
        print(f"\nSaved model parameters to {path}")

    def load(self, path):
        """ Load model parameters """

        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint["q_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"\nLoaded model parameters from {path}")

    def eval(self, episodes, render=False):
        """ Evaluates model performance """

        print(f"\nEvaluating model for {episodes} episodes ...\n")
        start_time = time.time()
        self.Q.eval()
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

                action = self._select_action(observation)
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

    model = DDPG(env)
    model.train(VERBOSE=True, PLOT_REWARDS=True)
    model.save(f"models/ddpg_torch_{env_name}.pt")
    # model.load(f"models/ddpg_torch_{env_name}.pt")
    model.eval(10, render=True)