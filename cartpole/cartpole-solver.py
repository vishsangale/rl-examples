import argparse
import math
from collections import namedtuple
from dataclasses import dataclass
from itertools import count

from PIL import Image
import random

import gym
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dqn import DQN

Transition = namedtuple('Tuple', ('state', 'action', 'next_state', 'reward'))


@dataclass
class Params:
    batch_size = 256
    gamma = 0.999
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = 200
    target_update = 10
    learning_rate = 0.01
    weight_decay = 0.01


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


def get_screen(env, device):
    # transpose to torch order CHW
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    _, height, width = screen.shape
    screen = screen[:, int(height * 0.4):int(height * 0.8)]

    view_width = int(width * 0.6)
    cart_location = get_cart_location(env, width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    screen = screen[:, :, slice_range]

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()
                        ])
    return resize(screen).unsqueeze(0).to(device)


def select_action(state, steps_done, policy_net, nr_actions, device):
    sample = random.random()
    threshold = Params.epsilon_end + (Params.epsilon_start - Params.epsilon_end) * math.exp(
        -1. * steps_done / Params.epsilon_decay)

    steps_done += 1

    if sample > threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(nr_actions)]], device=device, dtype=torch.long)


def main(args):
    random.seed(13)
    torch.manual_seed(13)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using `{device}` device")

    env = gym.make('CartPole-v0').unwrapped
    env.reset()

    # plt.figure()
    #
    # plt.imshow(get_screen(env, device).cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
    # plt.title("Example screen")
    # plt.show()

    screen = get_screen(env, device)
    _, _, height, width = screen.shape

    nr_actions = env.action_space.n

    policy_net = DQN(height, width, nr_actions).to(device)
    target_net = DQN(height, width, nr_actions).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=Params.learning_rate, weight_decay=Params.weight_decay)

    memory = ReplayMemory(10000)

    steps_done = 0

    episode_durations = []
    writer = SummaryWriter('/home/vsangale/workspace/tb-runs')

    nr_episodes = 300

    for i_episode in range(nr_episodes):
        env.reset()

        last_screen = get_screen(env, device)
        curr_screen = get_screen(env, device)

        state = curr_screen - last_screen

        for t in count():
            action = select_action(state, steps_done, policy_net, nr_actions, device)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # new state
            last_screen = curr_screen
            curr_screen = get_screen(env, device)
            if not done:
                next_state = curr_screen - last_screen
            else:
                next_state = None

            # transition
            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model(optimizer, policy_net, target_net, memory, device)

            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations, writer, t)
                break
        if i_episode % Params.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print("Training complete")
    env.render()
    env.close()
    writer.close()
    plt.show()


def plot_durations(episode_durations, writer, t):
    plt.figure(2)
    plt.clf()

    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    writer.add_scalar('Duration', episode_durations[-1], t)

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        current_mean = means.data[-1].item()
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
        writer.add_scalar('Duration Mean', current_mean, t)
        # print(means.numpy())

    plt.pause(0.001)


def optimize_model(optimizer, policy_net, target_net, memory, device):
    if len(memory) < Params.batch_size:
        return

    transitions = memory.sample(Params.batch_size)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(Params.batch_size, device=device)

    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * Params.gamma) + reward_batch

    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    for params in policy_net.parameters():
        params.grad.data.clamp_(-1, 1)

    optimizer.step()


def parse_arguments():
    parser = argparse.ArgumentParser(description='DQN algorithm for CartPole-v0 using OpenAI Gym')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
