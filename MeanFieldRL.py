from Configurations import *
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch
from copy import deepcopy
from torch.optim import Adam
import torch
import Utils
import pandas as pd
from Q_neural_network import Q


class MFQ:
    def __init__(self):
        self.num_of_agent = config_num_of_agents
        self.Q_network = Q()
        self.optimizer = Adam(self.Q_network.parameters(), lr=config_training_lr)

        # experience replay buffer
        self.buffer = {
            "states": np.zeros((config_buffer_capacity, 3 * self.num_of_agent + config_dim_average_action + 1 + 1)),
            "action": np.zeros((config_buffer_capacity, 1)),
            "next_states": np.zeros((config_buffer_capacity, 3 * self.num_of_agent + config_dim_average_action + 1+ 1)),
            "rewards": np.zeros((config_buffer_capacity, 1))}
        self.buffer_state = False
        self.pointer = 0

        # cuda usage
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.Q_network.cuda()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor

        torch.cuda.set_device(0)

    def add_experience(self, positions, average_actions, connected_state, action, next_positions, next_average_actions,
                       next_connected_state, rewards, remain_list):
        for i in remain_list:
            self.buffer["states"][self.pointer] = deepcopy(
                np.concatenate([positions.reshape(3 * self.num_of_agent), average_actions[i, :], [connected_state],[i]],
                               -1))
            self.buffer["action"][self.pointer] = deepcopy(action[i])
            self.buffer["next_states"][self.pointer] = deepcopy(
                np.concatenate(
                    [next_positions.reshape(3 * self.num_of_agent), next_average_actions[i, :], [next_connected_state],[i]],
                    -1)
            )
            self.buffer["rewards"][self.pointer] = deepcopy(rewards[i])
            if self.pointer < config_buffer_capacity - 1:
                self.pointer += 1
            else:
                self.pointer = 0
                self.buffer_state = True

    def __sample_experience(self):
        samples = {
            "states": np.zeros((config_training_batch, 3 * self.num_of_agent + config_dim_average_action + 1 + 1)),
            "action": np.zeros((config_training_batch, 1)),
            "next_states": np.zeros((config_training_batch, 3 * self.num_of_agent + config_dim_average_action + 1 + 1)),
            "rewards": np.zeros((config_training_batch, 1))}
        sample_list = Utils.random_sampling()
        counter = 0
        for i in sample_list:
            samples["states"][counter] = deepcopy(self.buffer["states"][i])
            samples["action"][counter] = deepcopy(self.buffer["action"][i])
            samples["next_states"][counter] = deepcopy(self.buffer["next_states"][i])
            samples["rewards"][counter] = deepcopy(self.buffer["rewards"][i])
            counter += 1
        return deepcopy(samples)

    def train(self):
        if self.buffer_state:
            samples = self.__sample_experience()
            states = torch.Tensor(samples["states"]).type(self.FloatTensor)
            action = torch.Tensor(samples["action"]).type(self.LongTensor)
            next_states = torch.Tensor(samples["next_states"]).type(self.FloatTensor)
            rewards = torch.Tensor(samples["rewards"]).type(self.FloatTensor)
            # 预测值
            q_eval = self.Q_network(states).gather(1, action)
            # 监督值（近似最优值）
            a = deepcopy(self.Q_network(next_states).detach())
            b = deepcopy(self.Boltzmann_policy(deepcopy(a)).type(self.FloatTensor))
            state_value_function = torch.mm(a, b.t())
            state_value_function = torch.diag(state_value_function).unsqueeze(-1)
            q_target = rewards + config_GAMMA * state_value_function

            loss = torch.nn.MSELoss()(q_eval, q_target.detach())

            # 训练三件套
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        else:
            print("Exploring ...")

    def target_policy(self, positions, average_actions, connection_state,remain_list):
        actions = np.zeros(self.num_of_agent, dtype=int)
        state = positions.reshape([3 * self.num_of_agent])
        connection_state = [int(connection_state)]
        for i in remain_list:
            a = np.concatenate([state, average_actions[i, :], connection_state], -1)
            a = torch.Tensor(a).type(self.FloatTensor)
            q_values = self.Q_network(a)
            actions[i] = deepcopy(self.Boltzmann_sample(q_values))
        return deepcopy(actions)

    def behavior_policy(self,positions, average_actions, connection_state,remain_list):
        actions = np.zeros(self.num_of_agent, dtype=int)
        state = positions.reshape([3 * self.num_of_agent])
        connection_state = [int(connection_state)]
        for i in remain_list:
            a = np.concatenate([state, average_actions[i, :], connection_state,[i]], -1)
            a = torch.Tensor(a).type(self.FloatTensor)
            q_values = self.Q_network(a)
            if np.random.uniform() <=1:
                actions[i] = deepcopy(self.Boltzmann_sample(q_values))
            else:
                actions[i] = np.random.randint(0,config_dim_action)
        return deepcopy(actions)

    def Boltzmann_policy(self, Q_values):
        prob = torch.Tensor(Q_values.size())
        for i in range(config_dim_action):
            prob[:, i] = torch.exp(-config_temperature * Q_values[:, i])
        for i in range(len(Q_values)):
            prob[i] /= torch.sum(prob[i])
        return prob

    def Boltzmann_sample(self, Q_values):
        prob = torch.zeros(config_dim_action)
        for i in range(config_dim_action):
            prob[i] = torch.exp(-config_temperature * Q_values[i])
        prob /= torch.sum(prob)
        sampling_assistant = torch.zeros(config_dim_action)
        for i in range(config_dim_action):
            if i == 0:
                sampling_assistant[i] = prob[0]
            else:
                sampling_assistant[i] = sampling_assistant[i - 1] + prob[i]

        sampling_assistant = sampling_assistant.detach().cpu().numpy()
        random_value = np.random.uniform(low=0.0, high=1.0)
        if random_value > sampling_assistant[config_dim_action - 2]:
            return config_dim_action - 1
        elif random_value <= sampling_assistant[0]:
            return 0
        for i in range(config_dim_action - 1):
            if sampling_assistant[i] < random_value <= sampling_assistant[i + 1]:
                return i+1

    def __change_to_one_hot(self, other_action):
        average_action = np.zeros(config_dim_average_action)
        for i in range(len(other_action)):
            temp = np.zeros(config_dim_average_action)
            temp[other_action[i]] = 1
            average_action += temp
        return deepcopy(average_action / len(other_action))

    def save_Q_function(self):
        torch.save(self.Q_network, 'optimal_Q/Q_network.pkl')

    def restore_Q_function(self):
        self.Q_network = torch.load("optimal_Q/Q_network.pkl")
        # cuda usage
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.Q_network.cuda()
