# Vanilla Policy Gradient in pytorch
# Atharv Sonwane <atharvs.twm@gmail.com>

# References -
# https://spinningup.openai.com/en/latest/algorithms/vpg.html

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Hyperparameters
EPOCHS = 200
EPISODES_PER_EPOCH = 4
POLICY_HIDDEN_LAYERS = 32
VALUE_HIDDEN_LAYERS = 32
GAMMA = 0.99
VALUE_FN_LEARNING_RATE = 1e-2
POLICY_LEARNING_RATE = 1e-3
MAX_TRAJ_LENGTH = 1000

# Model
class VPG:
    def __init__(
        self,
        env,
        policy_hidden_layers=POLICY_HIDDEN_LAYERS,
        value_hidden_layers=VALUE_HIDDEN_LAYERS,
    ):

        self.env = env
        if self.env.unwrapped.spec is not None:
            self.env_name = self.env.unwrapped.spec.id
        else:
            self.env_name = self.env.unwrapped.__class__.__name__
        self.policy = (
            nn.Sequential(
                nn.Linear(env.observation_space.shape[0], policy_hidden_layers),
                nn.Dropout(p=0.6),
                nn.ReLU(),
                nn.Linear(policy_hidden_layers, env.action_space.n),
                nn.Softmax(dim=0),
            )
            .to(device)
            .to(dtype)
        )
        self.value = (
            nn.Sequential(
                nn.Linear(env.observation_space.shape[0], value_hidden_layers),
                nn.Dropout(p=0.6),
                nn.ReLU(),
                nn.Linear(value_hidden_layers, 1),
            )
            .to(device)
            .to(dtype)
        )

    def train(
        self,
        epochs=EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        value_lr=VALUE_FN_LEARNING_RATE,
        policy_lr=POLICY_LEARNING_RATE,
        gamma=GAMMA,
        max_traj_length=MAX_TRAJ_LENGTH,
        RENDER=False,
        PLOT_REWARDS=True,
        VERBOSE=False,
    ):
        """ Trains both policy and value networks """

        hp = locals()
        print(
            f"\nTraining model on {self.env_name} | "
            f"Observation Space: {self.env.observation_space} | "
            f"Action Space: {self.env.action_space}\n"
            f"Hyperparameters: \n{hp}\n"
        )
        start_time = time.time()
        self.policy.train()
        self.value.train()
        value_optim = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        policy_optim = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        rewards = []
        e = 0

        for epoch in range(epochs):

            epoch_rewards = []

            value_loss = torch.tensor(
                [[0.0]], device=device, dtype=dtype, requires_grad=True
            )
            policy_loss = torch.tensor(
                [[0.0]], device=device, dtype=dtype, requires_grad=True
            )

            # Sample trajectories
            for _ in range(episodes_per_epoch):

                # initialise tracking variables
                observation = self.env.reset()
                observation = torch.tensor(observation, device=device, dtype=dtype)
                traj_observations = [observation]
                traj_obs_values = [self.value(observation)]
                traj_actions = []
                traj_action_distributions = []
                traj_rewards = []
                done = False
                e += 1

                # run for single trajectory
                for i in range(max_traj_length):
                    if RENDER and (
                        e == 0 or (e % ((epochs * episodes_per_epoch) / 10)) == 0
                    ):
                        self.env.render()

                    action_probs = self.policy(observation)
                    action_distribution = torch.distributions.Categorical(action_probs)
                    action = action_distribution.sample()
                    traj_actions.append(action)
                    traj_action_distributions.append(action_distribution)

                    observation, reward, done, _ = self.env.step(action.item())

                    observation = torch.tensor(observation, device=device, dtype=dtype)
                    reward = torch.tensor(reward, device=device, dtype=dtype)
                    traj_observations.append(observation)
                    traj_rewards.append(reward)
                    if not done:
                        traj_obs_values.append(self.value(observation))
                    if done:
                        break

                # update value function loss using sum of discounted rewards to go
                disc_rewards = []
                r = 0.0
                for reward in traj_rewards[::-1]:
                    r = reward + gamma * r
                    disc_rewards.insert(0, r)
                disc_rewards = torch.tensor(disc_rewards, device=device, dtype=dtype)
                disc_rewards = (disc_rewards - disc_rewards.mean()) / (
                    disc_rewards.std() + np.finfo(np.float32).eps
                )
                traj_obs_values = torch.tensor(
                    traj_obs_values, device=device, dtype=dtype
                )
                value_loss = value_loss + nn.MSELoss()(disc_rewards, traj_obs_values)

                # update policy function loss using advatage estimate with baseline
                for j in range(len(traj_rewards) - 1):
                    with torch.no_grad():
                        # advantage = (
                        #     traj_rewards[j] + gamma * traj_obs_values[j + 1]
                        # ) - traj_obs_values[j]
                        # advantage = (
                        #     traj_rewards[j]
                        #     + gamma * self.value(traj_observations[j + 1])
                        # ) - self.value(traj_observations[j])
                        # advantage = disc_rewards[j] - self.value(traj_observations[j])
                        advantage = (
                            disc_rewards[j] - traj_obs_values[j]
                        )  # This configuration performed better

                    policy_loss = (
                        policy_loss
                        + -1
                        * traj_action_distributions[j].log_prob(traj_actions[j])
                        * advantage
                    )

                epoch_rewards.append(sum(traj_rewards))

            # Fit value function
            value_loss = value_loss / episodes_per_epoch
            value_loss.backward()
            value_optim.step()
            value_optim.zero_grad()

            # Update policy
            policy_loss /= episodes_per_epoch
            policy_loss.backward()
            policy_optim.step()
            policy_optim.zero_grad()

            # Log rewards and losses to console
            avg_episode_reward = np.mean(epoch_rewards[-episodes_per_epoch:])
            rewards.append(avg_episode_reward)

            if VERBOSE and (epoch == 0 or ((epoch + 1) % (epochs / 10)) == 0):
                print(
                    f"Epoch {epoch+1}: Average Episodic Reward = {avg_episode_reward:.2f} |"
                    f" Value Loss = {value_loss.item():.2f} |"
                    f" Policy Loss = {policy_loss.item():.2f}"
                )

        self.env.close()
        print(f"\nTraining Completed in {(time.time() - start_time):.2f} seconds")
        if PLOT_REWARDS:
            plt.plot(rewards)
            plt.savefig(
                f"./plots/{self.__class__.__name__}_{self.env_name}_reward_plot.png"
            )

    def save(self, path=None):
        """ Save model parameters """
        if path is None:
            path = f"./models/{self.__class__.__name__}_{self.env_name}.pt"
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value.state_dict(),
            },
            path,
        )
        print(f"\nSaved model parameters to {path}")

    def load(self, path=None):
        """ Load model parameters """
        if path is None:
            path = f"./models/{self.__class__.__name__}_{self.env_name}.pt"
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])
        print(f"\nLoaded model parameters from {path}")

    def eval(self, episodes, render=False):
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

                action_probs = self.policy(observation)
                action_distribution = torch.distributions.Categorical(action_probs)
                action = action_distribution.sample()

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

        env.close()
        print(f"\nAverage Reward for an episode = {np.mean(rewards):.2f}")
        print(f"Evaluation Completed in {(time.time() - start_time):.2f} seconds")


if __name__ == "__main__":

    import gym

    env = gym.make("CartPole-v1")
    # env = gym.make("LunarLander-v2")

    # from pybullet_envs import bullet
    # env = bullet.racecarGymEnv.RacecarGymEnv(renders=False, isDiscrete=True)

    model = VPG(env)
    model.train(VERBOSE=True, PLOT_REWARDS=True)
    model.save()
    # model.load()
    model.eval(10)
