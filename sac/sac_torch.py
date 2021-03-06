# Soft Actor Critic in pytorch
# Atharv Sonwane <atharvs.twm@gmail.com>

# References -
# https://arxiv.org/pdf/1801.01290.pdf
# https://arxiv.org/pdf/1812.05905.pdf
# https://spinningup.openai.com/en/latest/algorithms/sac.html

import copy
import random
import traceback
from collections import deque, namedtuple
from datetime import datetime
from itertools import count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

dtype = torch.double

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
        self.common_model = (
            nn.Sequential(
                nn.Linear(s_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            .to(device)
            .to(dtype)
        )
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
            logprob -= 2 * (np.log(2) - logprob - nn.functional.softplus(-2 * logprob))

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
        self.model = (
            nn.Sequential(
                nn.Linear(s_dim + a_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            .to(device)
            .to(dtype)
        )

    def forward(self, s, a):
        x = torch.cat([s.view(-1, self.s_dim), a.view(-1, self.a_dim)], dim=1)
        return self.model(x)


# Model
class SAC:
    def __init__(self, env, q_hidden_dim, policy_hidden_dim):

        self.env = env
        if self.env.unwrapped.spec is not None:
            self.env_name = env.unwrapped.spec.id
        else:
            self.env_name = env.unwrapped.__class__.__name__
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
            self.observation_shape[0],
            policy_hidden_dim,
            self.action_shape[0],
            self.action_max,
            self.action_min,
        )

        self.entropy_temp_log = torch.zeros(
            1, device=device, dtype=dtype, requires_grad=True
        )
        self.entropy_temp = torch.exp(self.entropy_temp_log).item()
        self.target_entropy = -1 * np.prod(self.action_shape)

    def _select_action(
        self, observation, select_after=0, step_count=0, deterministic=False
    ):
        if step_count < select_after:
            action = torch.tensor(
                env.action_space.sample(), device=device, dtype=dtype
            ).unsqueeze(0)
        else:
            with torch.no_grad():
                action, _ = self.policy.sample_action(
                    observation, get_logprob=False, deterministic=deterministic
                )

        return action

    def _update(
        self,
        replay_buffer,
        batch_size,
        q_optimizers,
        policy_optimizer,
        gamma,
        polyak_const,
        entropy_temp_optimizer,
        grad_norm_clip,
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
            next_state_q_val = torch.min(
                *[Q(sample.next_observation, next_action) for Q in self.target_Qs]
            )
            # Compute targets using entropy objective
            target_q_vals = sample.reward + gamma * (
                next_state_q_val - self.entropy_temp * logprob
            ) * (~sample.done)

        # Update all the Q networks
        q_loss = 0.0
        for Q, target_Q, q_optimizer in zip(self.Qs, self.target_Qs, q_optimizers):

            # Compute the current Q values for each state action pair in batch
            q_vals = Q(sample.observation, sample.action)

            # Compute the q loss, backpropogate and update the gradients
            q_optimizer.zero_grad()
            curr_q_loss = nn.MSELoss()(target_q_vals, q_vals)
            curr_q_loss.backward()
            nn.utils.clip_grad_norm_(Q.parameters(), grad_norm_clip)
            q_optimizer.step()
            q_loss += float(curr_q_loss)

        # Turn off gradient calculation for Q networks to save compute during policy update
        for Q in self.Qs:
            for p in Q.parameters():
                p.requires_grad = False

        # Compute the policy loss, backpropogate and update the gradients
        policy_optimizer.zero_grad()
        action, logprob = self.policy.sample_action(sample.observation)
        min_q_val = torch.min(*[Q(sample.observation, action) for Q in self.Qs])
        policy_loss = -1 * torch.mean(min_q_val - self.entropy_temp * logprob)
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), grad_norm_clip)
        policy_optimizer.step()

        # Turn on gradient calculation for Q networks after policy update
        for Q in self.Qs:
            for p in Q.parameters():
                p.requires_grad = True

        # Tune entropy regularization temperature
        # entropy_temp_optimizer.zero_grad()
        # entropy_temp_loss = -1 * torch.mean(
        #     self.entropy_temp_log * (logprob + self.target_entropy).detach()
        # )
        # entropy_temp_loss.backward()
        # entropy_temp_optimizer.step()
        self.entropy_temp = 0.02  # torch.exp(self.entropy_temp_log).item()

        # Update target q networks with polyak averaging
        for Q, target_Q in zip(self.Qs, self.target_Qs):
            with torch.no_grad():
                for p_target, p in zip(target_Q.parameters(), Q.parameters()):
                    p_target.data.mul_(polyak_const)
                    p_target.data.add_((1 - polyak_const) * p.data)

        return q_loss, policy_loss.item()

    def train(self, args):
        """ Trains q and policy network """
        start_time = datetime.now()
        if args.xpid == None:
            logdir = Path(args.logdir).joinpath(
                f"{self.__class__.__name__}-{self.env_name}-{start_time:%d%m%y-%H%M%S}"
            )
        else:
            logdir = Path(args.logdir).joinpath(f"{args.xpid}")
        logdir.mkdir(parents=True)
        with open(logdir.joinpath("hyperparameters.txt"), "w+") as f:
            f.write(f"Start time: {start_time:%d%m%y-%H%M%S}\n")
            f.write(f"{args}")
        print(
            f"\nStarting at {start_time:%d-%m-%y %H:%M:%S}"
            f"\nTraining model on {self.env_name} | "
            f"Observation Space: {self.env.observation_space} | "
            f"Action Space: {self.env.action_space}"
            f"\nLogging to {logdir}"
            f"\nHyperparameters: \n{args}\n"
        )
        self.policy.train()
        for Q in self.Qs:
            Q.train()
        q_optimizers = [torch.optim.Adam(Q.parameters(), lr=args.q_lr) for Q in self.Qs]
        policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.policy_lr)
        entropy_temp_optimizer = torch.optim.Adam(
            [self.entropy_temp_log], lr=args.temp_lr
        )
        replay_buffer = deque(maxlen=args.max_buffer_size)
        rewards = []
        step_count = 0

        try:
            for episode in count():
                observation = self.env.reset()
                observation = torch.tensor(
                    observation, device=device, dtype=dtype
                ).unsqueeze(0)
                done = torch.tensor([False], device=device, dtype=torch.bool).unsqueeze(
                    0
                )
                episode_rewards = []

                for _ in range(args.max_traj_length):
                    if args.mode.lower() == "train_render":
                        self.env.render()
                    step_count += 1

                    action = self._select_action(
                        observation,
                        select_after=args.act_after,
                        step_count=0,
                        deterministic=False,
                    )
                    next_observation, reward, done, _ = self.env.step(action[0])
                    episode_rewards.append(float(reward))
                    next_observation = torch.tensor(
                        next_observation, device=device, dtype=dtype
                    ).unsqueeze(0)
                    reward = torch.tensor(
                        [reward], device=device, dtype=dtype
                    ).unsqueeze(0)
                    done = torch.tensor(
                        [done], device=device, dtype=torch.bool
                    ).unsqueeze(0)

                    transition = Transition(
                        observation, action, reward, next_observation, done
                    )
                    replay_buffer.append(transition)
                    observation = next_observation

                    # Update the Deep Q Network if sufficient transitions available every interval
                    if (
                        step_count >= args.update_after
                        and step_count % args.update_every == 0
                    ):
                        q_loss, policy_loss = self._update(
                            replay_buffer,
                            args.batch_size,
                            q_optimizers,
                            policy_optimizer,
                            args.discounting,
                            args.polyack,
                            entropy_temp_optimizer,
                            args.grad_norm_clip,
                        )

                    if args.checkpoint_every is not None:
                        if (
                            step_count >= args.update_after
                            and step_count % args.checkpoint_every == 0
                        ):
                            self.save(logdir.joinpath(f"model_{step_count}.pt"))

                    if done or step_count == args.timesteps:
                        break

                if step_count == args.timesteps:
                    break

                # Log rewards and losses
                total_episode_reward = sum(episode_rewards)
                rewards.append(total_episode_reward)
                if args.verbose:
                    print(
                        f"Episode {episode+1} (Step Count = {step_count}) | Reward = {total_episode_reward:.2f} | ",
                        end="",
                    )
                    if step_count >= args.update_after:
                        print(
                            f" Q Loss = {q_loss:.2f} | Policy Loss = {policy_loss:.2f} | {self.entropy_temp}"
                        )
                    else:
                        print("Collecting Experience")

        except KeyboardInterrupt:
            print(f"Training interrupted by user\n")

        except Exception as e:
            print(f"Training interrupted by \n{e}\n")
            traceback.print_exc()
            raise e

        finally:
            self.env.close()
            print(
                f"\nTraining Completed in {(datetime.now() - start_time).seconds} seconds"
            )
            plt.plot(rewards)
            plt.title(f"Training {self.__class__.__name__} on {self.env_name}")
            plt.xlabel("Episodes")
            plt.ylabel("Rewards")
            self.save(logdir.joinpath("model.pt"))
            plt.savefig(logdir.joinpath(f"rewards_plot.png"))
            # latest_path = Path(args.logdir).joinpath("latest")
            # if latest_path.exists():
            #     latest_path.unlink()
            # latest_path.symlink_to(logdir, target_is_directory=True)

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
        self.Qs[0].load_state_dict(checkpoint["q1_state_dict"])
        self.Qs[1].load_state_dict(checkpoint["q2_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"\nLoaded model parameters from {path}")

    def test(self, args, deterministic=True):
        """ Evaluates model performance """
        start_time = datetime.now()
        if args.xpid == None:
            logdir = Path(args.logdir).joinpath("latest")
        else:
            logdir = Path(args.logdir).joinpath(f"{args.xpid}")
        assert (
            args.test_episodes is not None
        ), "test_episodes needs to be specified for testing"
        print(
            f"\nStarting at {start_time:%d-%m-%y %H:%M:%S}"
            f"\Testing model on {self.env_name} for {args.test_episodes} epsidoes"
        )
        self.load(logdir.joinpath("model.pt"))
        self.policy.eval()
        rewards = []

        for episode in range(args.test_episodes):

            observation = self.env.reset()
            observation = torch.tensor(observation, device=device, dtype=dtype)
            done = False
            episode_rewards = []

            while not done:
                if args.mode.lower() == "test_render":
                    self.env.render()

                action = self._select_action(observation, deterministic=deterministic)
                next_observation, reward, done, _ = self.env.step(action.detach())
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
        print(
            f"Evaluation Completed in {(datetime.now() - start_time).seconds} seconds"
        )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="Soft Actor Critic")

    parser.add_argument(
        "--env", "-e", type=str, default="Pendulum-v0", help="Gym environment."
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="train",
        choices=["train", "test", "train_render", "test_render"],
        help="Training or test mode.",
    )
    parser.add_argument("--xpid", default=None, help="Experiment id (default: None).")

    parser.add_argument(
        "--timesteps",
        "-t",
        default=10_000,
        type=int,
        metavar="T",
        help="Total environment steps to train for.",
    )
    parser.add_argument(
        "--test_episodes", default=None, type=int, help="Total episodes to test for."
    )
    parser.add_argument(
        "--max_traj_length",
        default=1000,
        type=int,
        help="Maximum number of timesteps in a trajectory.",
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, metavar="B", help="Learner batch size."
    )
    parser.add_argument(
        "--max_buffer_size",
        default=5000,
        type=int,
        help="Maximum size of replay buffer.",
    )
    parser.add_argument(
        "--update_after",
        default=500,
        type=int,
        help="Timesteps to start updating after.",
    )
    parser.add_argument(
        "--update_every", default=1, type=int, help="Timsteps interval to update after."
    )
    parser.add_argument(
        "--act_after",
        default=1000,
        type=int,
        help="Timsteps to start selecting action from policy after.",
    )

    parser.add_argument(
        "--policy_lr", default=1e-3, type=float, help="Policy learning rate."
    )
    parser.add_argument(
        "--q_lr", default=1e-3, type=float, help="Q function learning rate."
    )
    parser.add_argument(
        "--temp_lr", default=5e-4, type=float, help="Entropy temperature learning rate."
    )
    parser.add_argument(
        "--discounting", default=0.99, type=float, help="Discounting factor."
    )
    parser.add_argument(
        "--polyack", default=0.995, type=float, help="Poly32ack averaging constant."
    )
    parser.add_argument(
        "--grad_norm_clip", default=20, type=float, help="Global gradient norm clip."
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_false",
        help="Enable to log progres to console.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=5000,
        help="Timesteps to checkpoint model every.",
    )
    parser.add_argument(
        "--logdir",
        default="~/logs/myrl/SAC/",
        help="Root dir where experiment data will be saved.",
    )
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA.")

    def make_env(flags):
        import gym

        env_name = flags.env
        render = flags.mode == "test_render" or flags.mode == "train_render"
        if "Bullet" in env_name:
            import pybullet_envs

            try:
                env = gym.make(env_name, isDiscrete=False, renders=render)
            except TypeError:
                env = gym.make(env_name, renders=render)
        else:
            env = gym.make(env_name)
        return env

    args = parser.parse_args()
    env = make_env(args)

    if args.disable_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode.lower() == "train":
        model = SAC(env, 32, 32)
        model.train(args)
    elif args.mode.lower() == "test" or args.mode.lower() == "test_render":
        model = SAC(env, 32, 32)
        model.test(args)
