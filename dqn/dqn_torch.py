# Deep Q Network in pytorch
# Atharv Sonwane <atharvs.twm@gmail.com>

# References -
# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import torch
import torch.nn as nn
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import time
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Hyperparameters
TIMESTEPS = 5000
BATCH_SIZE = 128
NETWORK_HIDDEN_LAYERS = 64
REPLAY_BUFFER_CAPACITY = 3000
EPSILON = 0.1
GAMMA = 0.9
LEARNING_RATE = 1e-2
MAX_TRAJ_LENGTH = 1000
MIN_BUFFER_SIZE = BATCH_SIZE
TARGET_UPDATE_INTERVAL = 1000
SAVE_FREQUENCY = None

# Transition tuple
Transition = namedtuple(
    "Transition", ("observation", "action", "reward", "next_observation", "done")
)

# Model
class DQN:
    def __init__(
        self,
        env,
        network_hidden_layers=NETWORK_HIDDEN_LAYERS,
        epsilon=EPSILON,
        replay_buffer_capacity=REPLAY_BUFFER_CAPACITY,
        target_update_interval=TARGET_UPDATE_INTERVAL
    ):

        self.env = env
        self.epsilon = epsilon
        self.target_update_interval = target_update_interval
        self.replay_buffer = deque(maxlen=replay_buffer_capacity)
        self.Q = self._make_network(network_hidden_layers)
        self.target_Q = self._make_network(network_hidden_layers)


    def _make_network(self, network_hidden_layers):
        return nn.Sequential(
                nn.Linear(self.env.observation_space.shape[0], network_hidden_layers),
                nn.ReLU(),
                nn.Linear(network_hidden_layers, self.env.action_space.n),
            ).to(device).to(dtype)

    def _select_action(self, observation):
        if np.random.random() < self.epsilon:
            action = torch.tensor(
                env.action_space.sample(), device=device, dtype=dtype
            )
        else:
            q_values = self.Q(observation)
            action = torch.argmax(q_values)
        return action.unsqueeze(0)

    def _update(self, batch_size, optimizer, gamma):

        sample = Transition(
            *[
                torch.cat(i)
                for i in [*zip(*random.sample(self.replay_buffer, batch_size))]
            ]
        )

        # Compute the target Q values for each state action pair in batch
        target_q_vals = sample.reward + gamma * self.target_Q(
            sample.next_observation
        ).detach().max(1).values.unsqueeze(1) * (~sample.done)

        # Compute the current Q values for each state action pair in batch
        q_vals = self.Q(sample.observation).gather(1, sample.action)

        # Compute the loss, backpropogate and update the gradients
        # loss = (target_q_vals - q_vals).pow(2).sum()
        # loss = nn.SmoothL1Loss()(target_q_vals, q_vals)
        loss = nn.MSELoss()(target_q_vals, q_vals)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # Update target network with polyak averaging
        # for p_target, p in zip(self.target_Q.parameters(), self.Q.parameters()):
        #     p_target.data = self.tao * p.data + (1 - self.tao) * p_target.data
        return loss.item()

    def train(
        self,
        episodes=EPISODES,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        min_buffer_size=MIN_BUFFER_SIZE,
        save_freq=SAVE_FREQUENCY,
        render=False,
        plot_rewards=True,
        VERBOSE=False,
        PLOT_REWARDS=False,
        max_traj_length=MAX_TRAJ_LENGTH,
    ):
        """ Trains deep q network """

        print(
            f"\nTraining model for {episodes} episodes with "
            f"batch size of {batch_size} ...\n"
        )
        start_time = time.time()

        self.Q.train()

        optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        rewards = []
        e = 0
        loss = 0.0
        step_count = 0

        if VERBOSE: print("Collecting experience ...")
        for episode in range(episodes):
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
                if render and (
                    episode == 0 or (episode % ((episodes + 1) / 10)) == 0
                ):
                    self.env.render()

                action = self._select_action(observation).unsqueeze(0).long()
                next_observation, reward, done, _ = self.env.step(action.item())

                #     # modify the reward
                # x, x_dot, theta, theta_dot = next_observation
                # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                # r2 = (
                #     env.theta_threshold_radians - abs(theta)
                # ) / env.theta_threshold_radians - 0.5
                # reward = r1 + r2
                
                episode_rewards.append(reward)
                next_observation = torch.tensor(
                    next_observation, device=device, dtype=dtype
                ).unsqueeze(0)
                reward = torch.tensor([reward], device=device, dtype=dtype).unsqueeze(0)
                done = torch.tensor([done], device=device, dtype=torch.bool).unsqueeze(0)
                transition = Transition(
                    observation, action, reward, next_observation, done
                )
                self.replay_buffer.append(transition)
                observation = next_observation

                # Update the Deep Q Network if sufficient transitions available
                if len(self.replay_buffer) >= min_buffer_size:
                    loss = self._update(batch_size, optimizer, gamma)
                    
                    # Update target Q network every specified numer of steps
                    if step_count % self.target_update_interval == 0:
                        self.target_Q.load_state_dict(self.Q.state_dict())

                if save_freq is not None:
                    if step_count % save_freq == 0:
                        model.save(f"models/models/ddpg_torch_{step_count}")

                if done:
                    break

            # Log rewards and losses
            total_episode_reward = sum(episode_rewards)
            rewards.append(total_episode_reward)

            if VERBOSE and len(self.replay_buffer) >= min_buffer_size:
                print(
                    f"Episode {episode+1}: Episode Reward = {total_episode_reward:.2f} |"
                    f" DQN Loss = {loss:.2f}"
                )

        print(f"\nTraining Completed in {(time.time() - start_time):.2f} seconds")
        if PLOT_REWARDS:
            plt.plot(rewards)
            plt.savefig("dqn_reward_plot.png")
        env.close()

    def save(self, path):
        """ Save model parameters """

        torch.save(
            {"dqn_state_dict": self.Q.state_dict()}, path,
        )
        print(f"\nSaved model parameters to {path}")

    def load(self, path):
        """ Load model parameters """

        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint["dqn_state_dict"])
        print(f"\nLoaded model parameters from {path}")

    def eval(self, episodes, render=False):
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
                if render:
                    self.env.render()

                action = self._select_action(observation).long()
                next_observation, reward, done, _ = self.env.step(action.item())
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
    # env_name = "CartPole-v1"
    # env_name = "LunarLander-v2"
    # env = gym.make(env_name)
    env = RacecarGymEnv(renders=False, isDiscrete=True); env_name = "RacecarBulletEnv"

    print(
        f"Env: {env_name} |"
        f" Observation Space: {env.observation_space} |"
        f" Action Space: {env.action_space}"
    )

    model = DQN(env)
    model.train(VERBOSE=True, PLOT_REWARDS=True)
    model.save(f"models/dqn_torch_bulletcar.pt")
    # model.load(f"dqn_torch_{env_name}.pt")
    model.eval(10, render=True)
