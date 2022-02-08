import random

import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple

from torch import FloatTensor
from torch.autograd import Variable

Tensor = FloatTensor

EPSILON = 0.9  # epsilon used for epsilon greedy approach
GAMMA = 0.9
TARGET_NETWORK_REPLACE_FREQ = 10  # How frequently target netowrk updates
BATCH_SIZE = 32
LR = 0.01  # learning rate

IMG_WIDTH = 320
IMG_HEIGHT = 240


class Net(nn.Module):
    def __init__(self, w, h, out_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 160, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(160)
        self.conv2 = nn.Conv2d(160, 320, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(320)
        self.conv3 = nn.Conv2d(320, 320, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(320)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = conv_w * conv_h * 320
        self.head = nn.Linear(linear_input_size, out_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 一层卷积
        x = F.relu(self.bn2(self.conv2(x)))  # 两层卷积
        x = F.relu(self.bn3(self.conv3(x)))  # 三层卷积
        return self.head(x.view(x.size(0), -1))  # 全连接层


class DQN:
    def __init__(self, memory_capacity=100):
        self.policy_net, self.target_net = Net(IMG_WIDTH, IMG_HEIGHT, 5), Net(IMG_WIDTH, IMG_HEIGHT, 5)

        # Define counter, memory size and loss function
        self.learn_step_counter = 0  # count the steps of learning process

        self.memory = []
        self.position = 0  # counter used for experience replay buff
        self.capacity = memory_capacity

        self.optimizer = th.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        """
        This function is used to make decision based upon epsilon greedy
        :param x:
        :return:
        """
        x = th.unsqueeze(th.FloatTensor(x), 0)  # add 1 dimension to input state x
        x = x.permute(0, 3, 2, 1)  # 把图片维度从[batch, height, width, channel] 转为[batch, channel, height, width]
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            # use epsilon-greedy approach to take action
            actions_value = self.policy_net.forward(x)
            # print(torch.max(actions_value, 1))
            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the action of cart.
            action = th.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, 5)

        return action

    def push_memory(self, old_observation, action, reward, observation):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(th.unsqueeze(th.FloatTensor(old_observation), 0),
                                                th.unsqueeze(th.FloatTensor(observation), 0),
                                                th.from_numpy(np.array([action])),
                                                th.from_numpy(np.array([reward], dtype='int64')))
        self.position = (self.position + 1) % self.capacity

    def get_sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def learn(self):
        print(f"DQN learn.")
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network, and how to implement
        # backward propagation.

        # update the target network every fixed steps
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_step_counter += 1

        transitions = self.get_sample(BATCH_SIZE)  # 抽样
        batch = Transition(*zip(*transitions))

        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        cur_frame = Variable(th.cat(batch.state))
        # convert long int type to tensor
        b_a = Variable(th.cat(batch.action))
        b_r = Variable(th.cat(batch.reward))
        nxt_frame = Variable(th.cat(batch.next_state))

        cur_frame = cur_frame.permute(0, 3, 2, 1)  # 当前帧图像数据，(w * h * c) -> (c * w * h)
        nxt_frame = nxt_frame.permute(0, 3, 2, 1)  # 下一帧图像数据，(w * h * c) -> (c * w * h)

        # calculate the Q value of state-action pair
        q_eval = self.policy_net(cur_frame).gather(1, b_a.unsqueeze(1))  # (batch_size, 1)

        # calculate the q value of next state
        q_next = self.target_net(nxt_frame).detach()  # detach from computational graph, don't back propagate
        # select the maximum q value
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step


Transition = namedtuple("Transition", ("state", "next_state", "action", "reward"))
