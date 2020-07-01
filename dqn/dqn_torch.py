# Deep Q Network in pytorch
# Atharv Sonwane <atharvs.twm@gmail.com>

# References -
# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import copy
import random
import time
from collections import deque, namedtuple
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Hyperparameters
TIMESTEPS = 5000
BATCH_SIZE = 128
NETWORK_HIDDEN_LAYERS = 32
EPSILON = 0.1
GAMMA = 0.9
MAX_BUFFER_SIZE = TIMESTEPS // 2
POLYAK_CONST = 0.995
LEARNING_RATE = 1e-2
MAX_TRAJ_LENGTH = 1000
MIN_BUFFER_SIZE = BATCH_SIZE

# Transition tuple
Transition = namedtuple(
    "Transition", ("observation", "action", "reward", "next_observation", "done")
)

# Model
class DQN:
    def __init__(
        self, env, network_hidden_layers=NETWORK_HIDDEN_LAYERS,
    ):

        self.env = env
        if self.env.unwrapped.spec is not None:
            self.env_name = self.env.unwrapped.spec.id
        else:
            self.env_name = self.env.unwrapped.__class__.__name__

        self.Q = self._make_network(network_hidden_layers)
        self.target_Q = copy.deepcopy(self.Q)
        for p in self.target_Q.parameters():
            p.requires_grad = False

    def _make_network(self, network_hidden_layers):
        return (
            nn.Sequential(
                nn.Linear(self.env.observation_space.shape[0], network_hidden_layers),
                nn.ReLU(),
                nn.Linear(network_hidden_layers, self.env.action_space.n),
            )
            .to(device)
            .to(dtype)
        )

    def _select_action(self, observation, eps=0):
        if np.random.random() < eps:
            action = torch.tensor(env.action_space.sample(), device=device, dtype=dtype)
        else:
            q_values = self.Q(observation)
            action = torch.argmax(q_values)
        return action.unsqueeze(0)

    def _update(self, replay_buffer, batch_size, optimizer, gamma, polyak_const):

        sample = Transition(
            *[torch.cat(i) for i in [*zip(*random.sample(replay_buffer, batch_size))]]
        )

        # Compute the target Q values for each state action pair in batch
        with torch.no_grad():
            target_q_vals = sample.reward + gamma * self.target_Q(
                sample.next_observation
            ).max(1).values.unsqueeze(1) * (~sample.done)

        # Compute the current Q values for each state action pair in batch
        q_vals = self.Q(sample.observation).gather(1, sample.action)

        # Compute the loss, backpropogate and update the gradients
        # loss = nn.SmoothL1Loss()(target_q_vals, q_vals)
        optimizer.zero_grad()
        loss = nn.MSELoss()(target_q_vals, q_vals)
        loss.backward()
        optimizer.step()

        # Update target network with polyak averaging
        with torch.no_grad():
            for p_target, p in zip(self.target_Q.parameters(), self.Q.parameters()):
                p_target.data.mul_(polyak_const)
                p_target.data.add_((1 - polyak_const) * p.data)

        return loss.item()

    def train(
        self,
        timesteps=TIMESTEPS,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        eps=EPSILON,
        batch_size=BATCH_SIZE,
        polyak_const=POLYAK_CONST,
        min_buffer_size=MIN_BUFFER_SIZE,
        max_buffer_size=MAX_BUFFER_SIZE,
        max_traj_length=MAX_TRAJ_LENGTH,
        RENDER=False,
        VERBOSE=False,
        PLOT_REWARDS=False,
        SAVE_FREQUENCY=None,
    ):
        """ Trains deep q network """
        hp = locals()
        print(
            f"\nTraining model on {self.env_name} | "
            f"Observation Space: {self.env.observation_space} | "
            f"Action Space: {self.env.action_space}\n"
            f"Hyperparameters: \n{hp}\n"
        )
        start_time = time.time()
        self.Q.train()
        optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        replay_buffer = deque(maxlen=max_buffer_size)
        rewards = []
        e = 0
        loss = 0.0
        step_count = 0

        if VERBOSE:
            print("Collecting experience ...")
        for episode in count():
            # initialise tracking variables
            observation = self.env.reset()
            observation = torch.tensor(
                observation, device=device, dtype=dtype
            ).unsqueeze(0)
            done = torch.tensor([False], device=device, dtype=torch.bool).unsqueeze(0)
            episode_rewards = []

            # run for single trajectory
            for _ in range(max_traj_length):
                step_count += 1
                if RENDER:
                    self.env.render()

                action = self._select_action(observation, eps).unsqueeze(0).long()
                next_observation, reward, done, _ = self.env.step(action.item())

                episode_rewards.append(float(reward))
                next_observation = torch.tensor(
                    next_observation, device=device, dtype=dtype
                ).unsqueeze(0)
                reward = torch.tensor([reward], device=device, dtype=dtype).unsqueeze(0)
                done = torch.tensor([done], device=device, dtype=torch.bool).unsqueeze(
                    0
                )
                transition = Transition(
                    observation, action, reward, next_observation, done
                )
                replay_buffer.append(transition)
                observation = next_observation

                # Update the Deep Q Network if sufficient transitions available
                if len(replay_buffer) >= min_buffer_size:
                    loss = self._update(
                        replay_buffer, batch_size, optimizer, gamma, polyak_const
                    )

                if SAVE_FREQUENCY is not None:
                    if (
                        len(replay_buffer) >= min_buffer_size
                        and step_count % (timesteps // SAVE_FREQUENCY) == 0
                    ):
                        self.save()

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
                if len(replay_buffer) >= min_buffer_size:
                    print(f" DNQ Loss = {loss:.2f}")
                else:
                    print("Collecting Experience")

        self.env.close()
        print(f"\nTraining Completed in {(time.time() - start_time):.2f} seconds")
        if PLOT_REWARDS:
            plt.plot(rewards)
            plt.title(f"Training {self.__class__.__name__} on {self.env_name}")
            plt.xlabel("Episodes")
            plt.ylabel("Rewards")
            plt.savefig(
                f"./plots/{self.__class__.__name__}_{self.env_name}_reward_plot.png"
            )

    def save(self, path=None):
        """ Save model parameters """
        if path is None:
            path = f"./models/{self.__class__.__name__}_{self.env_name}.pt"
        torch.save(
            {f"dqn_state_dict": self.Q.state_dict()}, path,
        )
        print(f"\nSaved model parameters to {path}")

    def load(self, path=None):
        """ Load model parameters """
        if path is None:
            path = f"./models/{self.__class__.__name__}_{self.env_name}.pt"
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint["dqn_state_dict"])
        print(f"\nLoaded model parameters from {path}")

    def eval(self, episodes, RENDER=False):
        """ Evaluates model performance """

        print(f"\nEvaluating model for {episodes} episodes ...\n")
        start_time = time.time()
        self.Q.eval()
        rewards = []

        for episode in range(episodes):

            observation = self.env.reset()
            observation = torch.tensor(observation, device=device, dtype=dtype)
            done = False
            episode_rewards = []

            while not done:
                if RENDER:
                    self.env.render()

                action = self._select_action(observation).long()
                next_observation, reward, done, _ = self.env.step(action.item())
                episode_rewards.append(float(reward))
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

        self.env.close()
        print(f"\nAverage Reward for an episode = {np.mean(rewards):.2f}")
        print(f"Evaluation Completed in {(time.time() - start_time):.2f} seconds\n")


if __name__ == "__main__":

    # import gym
    # env = gym.make("CartPole-v1")
    # env = gym.make("LunarLander-v2")

    from pybullet_envs import bullet

    env = bullet.racecarGymEnv.RacecarGymEnv(renders=False, isDiscrete=True)

    model = DQN(env)
    model.train(VERBOSE=True, PLOT_REWARDS=True, SAVE_FREQUENCY=10)
    model.save()
    # model.load()
    model.eval(10, RENDER=True)
