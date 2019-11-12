import math
import os
import pickle
import random
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from env_wrapper import atari_wrapper
from additional_functions import TernaryTanh
from torch.autograd import Variable
import tools as tl
import torch.nn.functional as F
from tools import plot_data
import logging

class GRUNet(nn.Module):
    """
    Gated Recurrent Unit Network(GRUNet) definition.
    """

    def __init__(self, input_size, gru_cells, total_actions):
        super(GRUNet, self).__init__()
        self.gru_units = gru_cells
        self.noise = False
        self.conv1 = nn.Conv2d(input_size, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 8, 3, stride=2, padding=1)

        self.input_ff = nn.Sequential(self.conv1, nn.ReLU(),
                                      self.conv2, nn.ReLU(),
                                      self.conv3, nn.ReLU(),
                                      self.conv4, nn.ReLU6())
        self.input_c_features = 8 * 5 * 5
        self.input_c_shape = (8, 5, 5)
        self.gru = nn.GRUCell(self.input_c_features, gru_cells)

        self.critic_linear = nn.Linear(gru_cells, 1)
        self.actor_linear = nn.Linear(gru_cells, total_actions)

        self.apply(tl.weights_init)
        self.actor_linear.weight.data = tl.normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = tl.normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)

    def forward(self, input, input_fn=None, hx_fn=None, inspect=False):
        input, hx = input
        c_input = self.input_ff(input)
        c_input = c_input.view(-1, self.input_c_features)
        input, input_x = input_fn(c_input) if input_fn is not None else (c_input, c_input)
        ghx = self.gru(input, hx)

        # Keep the noise during both training as well as evaluation
        # c_input = gaussian(c_input, self.training, mean=0, std=0.05, one_sided=True)
        # c_input = tl.uniform(c_input, self.noise, low=-0.01, high=0.01, enforce_pos=True)
        # ghx = tl.uniform(ghx, self.noise, low=-0.01, high=0.01)

        hx, bhx = hx_fn(ghx) if hx_fn is not None else (ghx, ghx)

        if inspect:
            return self.critic_linear(hx), self.actor_linear(hx), hx, (ghx, bhx, c_input, input_x)
        else:
            return self.critic_linear(hx), self.actor_linear(hx), hx

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.gru_units)

    def get_action_linear(self, state):
        return self.actor_linear(state)

    def transact(self, o_x, hx):
        hx = self.gru(o_x, hx)
        return hx


class ConvObsQBNet(nn.Module):
    def __init__(self, channels, x_features):
        super(ConvObsQBNet, self).__init__()

        self.noise = False


        # self.input_ff = nn.Sequential()
        self.qbn_input_size = 8 * 5 * 5
        self.latent_size = x_features
        f1 = int(8 * x_features)
        self.conv_encoder = nn.Sequential(nn.Conv2d(channels, 32, 3, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 16, 3, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(16, 8, 3, stride=2, padding=1),
                                     nn.ReLU6())

        self.linear_encoder = nn.Sequential(nn.Linear(self.qbn_input_size, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, x_features),
                                     TernaryTanh())

        self.linear_decoder = nn.Sequential(nn.Linear(x_features, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, self.qbn_input_size),
                                     nn.ReLU6())

        self.conv_decoder = nn.Sequential(nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(32, channels, 3, stride=2, padding=1, output_padding=1),
                                     nn.Sigmoid()
                                     )

    def forward(self, x):
        conv_encoded = self.conv_encoder(x)
        # linear_encoder_input = conv_encoded.view(-1, self.qbn_input_size)
        linear_encoder_input = torch.reshape(conv_encoded, (x.shape[0], self.qbn_input_size))
        linear_encoded = self.linear_encoder(linear_encoder_input)
        linear_decoded = self.linear_decoder(linear_encoded)
        # conv_decoder_input = torch.reshape(linear_decoded, (1, 8, 5, 5))
        conv_decoder_input = torch.reshape(linear_decoded, (x.shape[0], 8, 5, 5))
        conv_decoded = self.conv_decoder(conv_decoder_input)
        return conv_decoded, linear_encoded

    def encode(self, x):
        conv_encoded = self.conv_encoder(x)
        linear_encoder_input = conv_encoded.view(-1, self.qbn_input_size)
        linear_encoded = self.linear_encoder(linear_encoder_input)
        return linear_encoded

    def decode(self, x):
        linear_decoded = self.linear_decoder(x)
        conv_decoder_input = torch.reshape(linear_decoded, (1, 8, 5, 5))
        conv_decoded = self.conv_decoder(conv_decoder_input)
        return conv_decoded

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.gru_units)

class ObsQBNet(nn.Module):
    """
    Quantized Bottleneck Network(QBN) for observation features.
    """

    def __init__(self, input_size, x_features):
        super(ObsQBNet, self).__init__()
        self.bhx_size = x_features
        f1 = int(8 * x_features)
        self.encoder = nn.Sequential(nn.Linear(input_size, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, x_features),
                                     TernaryTanh())

        self.decoder = nn.Sequential(nn.Linear(x_features, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, input_size),
                                     nn.ReLU6())

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


def generate_bottleneck_data(net, env, episodes, save_path, cuda=False, eps=(0, 0), max_steps=None):
    """
    Generating bottleneck data for the given network.

    :param net: given network
    :param env: given environment
    :param episodes: number of episodes
    :param save_path: path to save data in
    :param cuda: check if cuda is available
    :param max_steps: maximum number of steps to take. used for exploration.
    :return: observation and hidden state bottleneck data
    """
    if os.path.exists(save_path):
        # unpickling after reading the file is efficient
        hx_train_data, hx_test_data, obs_train_data, obs_test_data = pickle.loads(open(save_path, "rb").read())
    else:
        logging.info('No Data Found @ path : {}'.format(save_path))
        logging.info('Generating BottleNeck Data..')
        bottleneck_data = {}
        hx_data, obs_data, action_data = [], [], []
        all_ep_rewards = []
        with torch.no_grad():
            for ep in range(episodes):
                done = False
                obs = env.reset()
                hx = Variable(net.init_hidden())
                ep_reward = 0
                act_count = 0
                exploration_start_step = random.choice(range(0, max_steps, int(0.02 * max_steps)))
                while not done:
                    # env.render()
                    obs = Variable(torch.Tensor(obs)).unsqueeze(0)
                    if cuda:
                        hx = hx.cuda()
                        obs = obs.cuda()
                    critic, logit, hx, (_, _, obs_c, _) = net((obs, hx), inspect=True)
                    if exploration_start_step >= act_count and random.random() < eps[ep % len(eps)]:
                        action = env.action_space.sample()
                    else:
                        prob = F.softmax(logit, dim=1)
                        action = int(prob.max(1)[1].data.cpu().numpy())

                    if action not in bottleneck_data:
                        bottleneck_data[action] = {'hx_data': [], 'obs_data': []}
                    bottleneck_data[action]['obs_data'].append(obs.data.cpu().numpy()[0].tolist())
                    bottleneck_data[action]['hx_data'].append(hx.data.cpu().numpy()[0].tolist())

                    obs, reward, done, info = env.step(action)
                    action_data.append(action)
                    act_count += 1
                    done = done if act_count <= max_steps else True

                    ep_reward += reward
                logging.info('episode:{} reward:{}'.format(ep, ep_reward))
                all_ep_rewards.append(ep_reward)
        logging.info('Average Performance:{}'.format(sum(all_ep_rewards) / len(all_ep_rewards)))

        hx_train_data, hx_test_data, obs_train_data, obs_test_data = [], [], [], []
        for action in bottleneck_data.keys():
            hx_train_data += bottleneck_data[action]['hx_data']
            hx_test_data += bottleneck_data[action]['hx_data']
            obs_train_data += bottleneck_data[action]['obs_data']
            obs_test_data += bottleneck_data[action]['obs_data']

            # logging.info('Action: {} Hx Data: {} Obs Data: {}'.format(action, len(np.unique(bottleneck_data[action]['hx_data'], axis=0).tolist()), len(np.unique(bottleneck_data[action]['obs_data'], axis=0).tolist())))

        obs_test_data = np.unique(obs_test_data, axis=0).tolist()
        hx_test_data = np.unique(hx_test_data, axis=0).tolist()

        random.shuffle(hx_train_data)
        random.shuffle(obs_train_data)
        random.shuffle(hx_test_data)
        random.shuffle(obs_test_data)

        pickle.dump((hx_train_data, hx_test_data, obs_train_data, obs_test_data), open(save_path, "wb"))

    logging.info('Data Sizes:')
    logging.info('Hx Train:{} Hx Test:{} Obs Train:{} Obs Test:{}'.format(len(hx_train_data), len(hx_test_data), len(obs_train_data), len(obs_test_data)))

    return hx_train_data, hx_test_data, obs_train_data, obs_test_data


def train(net, obs_train_data, obs_test_data, optimizer, model_path, plot_dir, batch_size, epochs, target_net, cuda=False, grad_clip=None, env=None, low=0, high=0.05, target_test_episodes=1, base_background=None):
    """
    Train the QBN

    :param net: given network
    :param data: given data to train the network on
    :param optimizer: optimizer method(Adam is preferred)
    :param model_path: path to where save the model
    :param plot_dir: path to where save the plots
    :param batch_size: batch size
    :param epochs: number of training epochs
    :param cuda: check if cuda is available
    :param grad_clip: max norm of the gradients
    :param env: environment
    :param low: lower bound of noise data
    :param high: upper bound of noise data
    :param target_test_episodes: number of episodes to test on
    :return: returns the trained model
    """
    mse_loss = nn.MSELoss().cuda() if cuda else nn.MSELoss()
    train_data, test_data = obs_train_data, obs_test_data

    min_loss_i, best_perf_i = None, None
    batch_loss_data, epoch_losses, test_losses, test_perf_data = [], [], [], []
    total_batches = math.ceil(len(train_data) / batch_size)

    for epoch in range(epochs):
        net.train()
        batch_losses = []
        random.shuffle(train_data)
        # for t_i in range(len(train_data)):
        for b_i in range(total_batches):
            batch_input = train_data[b_i:b_i + batch_size]
            # input = train_data[t_i]
            target = Variable(torch.FloatTensor(batch_input))
            # batch_input = torch.FloatTensor(batch_input)
            batch_input = Variable(torch.FloatTensor(batch_input), requires_grad=True)

            if cuda:
                batch_input, target = batch_input.cuda(), target.cuda()

            if base_background is None:
                batch_output, _ = net(batch_input)
            else:
                batch_delta_output, _ = net(batch_input)
                batch_output = Variable(torch.FloatTensor(base_background)) + batch_delta_output

            optimizer.zero_grad()
            loss = mse_loss(batch_output, target)
            loss.backward()
            batch_losses.append(loss.item())
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            # print('epoch: %d step: %d loss: %f' % (epoch, t_i, loss.item()))
            print('epoch: %d batch: %d loss: %f' % (epoch, b_i, loss.item()))

        batch_loss_data += batch_losses
        epoch_losses.append(round(np.average(batch_losses), 5))
        test_losses.append(round(test(net, test_data, len(test_data), cuda=cuda), 5))
        # test_perf = test_with_env(target_net, env, target_test_episodes, cuda=cuda)
        # test_perf_data.append(test_perf)

        # if (best_perf_i is None) or (test_perf_data[best_perf_i] <= test_perf_data[-1]) or test_perf_data[
        #     -1] == env.spec.reward_threshold:
        #     torch.save(net.state_dict(), model_path)
        #     print('Bottle Net Model Saved!')
        # if (best_perf_i is None) or (test_perf_data[best_perf_i] < test_perf_data[-1]):
        #     best_perf_i = len(test_perf_data) - 1
        #     print('Best Perf i updated')
        if (min_loss_i is None) or (test_losses[min_loss_i] > test_losses[-1]):
            min_loss_i = len(test_losses) - 1
            torch.save(net.state_dict(), model_path)
            print('min_loss_i updated')
            print('Bottle Net Model Saved!')

        # plot_data(verbose_data_dict(test_losses, epoch_losses, batch_loss_data, test_perf_data), plot_dir)
        plot_data(verbose_data_dict(test_losses, epoch_losses, batch_loss_data), plot_dir, env_name)
        # print('epoch: %d test loss: %f best perf i: %d min loss i: %d' % (epoch, test_losses[-1], best_perf_i,
        print('epoch: %d test loss: %f min loss i: %d' % (epoch, test_losses[-1], min_loss_i))

        if np.isnan(batch_losses[-1]):
            print('Batch Loss: Nan')
            break
        if ((len(test_losses) - 1 - min_loss_i) > 50) or (test_losses[-1] == 0):
            print('Test Loss hasn\'t improved in last 50 epochs' if test_losses[-1] != 0 else 'Zero Test Loss!!')
            print('Stopping!')
            break

    torch.save(net.state_dict(), model_path.replace("pongD_gru_model.p", "last_model.p"))

    net.load_state_dict(torch.load(model_path))
    return net


def test(net, data, batch_size, cuda=False):
    """
    Test the trained network.

    :param net: given network
    :param data: given data to test the network on
    :param batch_size: batch size
    :param cuda: check if cuda is available
    :return: test performance
    """
    mse_loss = nn.MSELoss().cuda() if cuda else nn.MSELoss()
    net.eval()
    batch_losses = []
    # total_batches = int(len(data) / batch_size)
    # if len(data) % batch_size != 0:
    #     total_batches += 1
    with torch.no_grad():
        for t_i in range(len(data)):
            # input = data[b_i:b_i + batch_size]
            input = data[t_i]
            # input = Variable(torch.FloatTensor(input))
            input = Variable(torch.FloatTensor(input).unsqueeze(0), requires_grad=True)
            target = Variable(torch.FloatTensor(input))
            if cuda:
                target, input = target.cuda(), input.cuda()
            batch_output, _ = net(input)
            loss = mse_loss(batch_output, target)
            batch_losses.append(float(loss.item()))

    return sum(batch_losses) / len(batch_losses)


def gru_nn_test(net, env, total_episodes, test_seeds=None, cuda=False, log=False, render=False, max_actions=5000):
    """
    Test the performance of the given network.

    :param net: trained Bottleneck GRU network
    :param env: environment
    :param total_episodes: number of episodes of testing
    :param test_seeds: test seeds
    :param cuda: check if cuda is available
    :param log: check to print out test log
    :param render: check to render environment
    :param max_actions: max number of actions
    :return: test performance on trained model
    """
    net.eval()
    total_reward = 0
    with torch.no_grad():
        for ep in range(total_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0
            ep_actions = []
            hx = Variable(net.init_hidden())
            all_observations = [obs]
            action_count = 0
            while not done:
                # if render:
                #     env.render()
                obs = Variable(torch.Tensor(obs)).unsqueeze(0)
                if cuda:
                    obs, hx = obs.cuda(), hx.cuda()
                critic, logit, hx = net((obs, hx))
                prob = F.softmax(logit, dim=1)
                action = int(prob.max(1)[1].data.cpu().numpy())
                obs, reward, done, _ = env.step(action)
                action_count += 1
                done = done if action_count <= max_actions else True
                ep_actions.append(action)
                # A quick hack to prevent the agent from stucking
                max_same_action = 5000
                if action_count > max_same_action:
                    actions_to_consider = ep_actions[-max_same_action:]
                    if actions_to_consider.count(actions_to_consider[0]) == max_same_action:
                        done = True
                ep_reward += reward
                if not done:
                    all_observations.append(obs)
            total_reward += ep_reward
            if log:
                print('Episode =>{} Score=> {} Actions=> {} ActionCount=> {}'.format(ep, ep_reward, ep_actions, action_count))
        return total_reward / total_episodes


def verbose_data_dict(test_loss, reconstruction_epoch_losses, reconstruction_batch_losses, regularized_epoch_losses=None, regularized_batch_losses=None):
    """
    Makes data(losses and performance) into a dictionary for the sake of data plotting.

    :param test_loss: test performance
    :param epoch_losses: MSE and CE epoch loss
    :param batch_losses: MSE and CE batch loss
    :param test_env_perf: performance on test environment
    :return: data info dictionary
    """
    if regularized_epoch_losses is not None and regularized_batch_losses is not None:
        data_dict = [
            {'title': "Test_Loss_vs_Epoch", 'data': test_loss, 'y_label': 'Loss(' + str(min(test_loss)) + ')',
             'x_label': 'Epoch'},
            {'title': "Reconstruction Loss_vs_Epoch", 'data': reconstruction_epoch_losses, 'y_label': 'Loss(' + str(min(reconstruction_epoch_losses)) + ')',
             'x_label': 'Epoch'},
            {'title': "Regularized Loss_vs_Epoch", 'data': regularized_epoch_losses,
             'y_label': 'Loss(' + str(min(regularized_epoch_losses)) + ')',
             'x_label': 'Epoch'},
            {'title': "Reconstruction Loss_vs_Batches", 'data': reconstruction_batch_losses, 'y_label': 'Loss(' + str(min(reconstruction_batch_losses)) + ')',
             'x_label': 'Batch'},
            {'title': "Regularized Loss_vs_Batches", 'data': regularized_batch_losses,
             'y_label': 'Loss(' + str(min(regularized_batch_losses)) + ')',
             'x_label': 'Batch'}
        ]
    else:
        data_dict = [
            {'title': "Test_Loss_vs_Epoch", 'data': test_loss, 'y_label': 'Loss(' + str(min(test_loss)) + ')',
             'x_label': 'Epoch'},
            {'title': "Construction Loss_vs_Epoch", 'data': reconstruction_epoch_losses,
             'y_label': 'Loss(' + str(min(reconstruction_epoch_losses)) + ')',
             'x_label': 'Epoch'},
            {'title': "Loss_vs_Batches", 'data': reconstruction_batch_losses, 'y_label': 'Loss(' + str(min(reconstruction_batch_losses)) + ')',
             'x_label': 'Batch'}
        ]

    return data_dict


def gather_base_image(bottleneck_data_path):
    hx_train_data, hx_test_data, obs_train_data, obs_test_data = pickle.loads(open(bottleneck_data_path, "rb").read())
    numpied_obs = np.array(obs_train_data)
    avg_base = np.mean(numpied_obs, axis=0)
    return avg_base.tolist()


if __name__ == '__main__':
    episodes = 2
    gru_size = 32
    bhx_size = 64
    ox_size = 100
    input_c_features = 8 * 5 * 5
    bn_episodes = 10
    # bn_episodes = 1
    num_epoch = 400
    # num_epoch = 20
    bottleneck_data_path = "./resources/pongD_bottleneck_data.p"
    generate_max_steps = 10000

    env_name = "PongDeterministic-v4"
    env = atari_wrapper(env_name)
    obs = env.reset()

    gru_net_path = "./resources/pongD_gru_model.p"
    gru_net = GRUNet(len(obs), gru_size, int(env.action_space.n))
    gru_net.load_state_dict(torch.load(gru_net_path))
    gru_net.noise = False
    gru_net.eval()

    ox_net_path = "./resources/pongD_obs_model.p"
    ox_net = ObsQBNet(gru_net.input_c_features, ox_size)
    ox_net.load_state_dict(torch.load(ox_net_path))
    ox_net.eval()

    conv_ox_net = ConvObsQBNet(len(obs), ox_size)

    hx_train_data, hx_test_data, obs_train_data, obs_test_data = generate_bottleneck_data(gru_net, env, bn_episodes, bottleneck_data_path, cuda=False, eps=(0, 0.3), max_steps=generate_max_steps)


    for name, param in gru_net.state_dict().items():
        if name == "conv1.weight":
            conv_ox_net.state_dict()["conv_encoder.0.weight"].data.copy_(param.data)
            conv_ox_net.conv_encoder[0].weight.requires_grad = False
        elif name == "conv1.bias":
            conv_ox_net.state_dict()["conv_encoder.0.bias"].data.copy_(param.data)
            conv_ox_net.conv_encoder[0].bias.requires_grad = False
        elif name == "conv2.weight":
            conv_ox_net.state_dict()["conv_encoder.2.weight"].data.copy_(param.data)
            conv_ox_net.conv_encoder[2].weight.requires_grad = False
        elif name == "conv2.bias":
            conv_ox_net.state_dict()["conv_encoder.2.bias"].data.copy_(param.data)
            conv_ox_net.conv_encoder[2].bias.requires_grad = False
        elif name == "conv3.weight":
            conv_ox_net.state_dict()["conv_encoder.4.weight"].data.copy_(param.data)
            conv_ox_net.conv_encoder[4].weight.requires_grad = False
        elif name == "conv3.bias":
            conv_ox_net.state_dict()["conv_encoder.4.bias"].data.copy_(param.data)
            conv_ox_net.conv_encoder[4].bias.requires_grad = False
        elif name == "conv4.weight":
            conv_ox_net.state_dict()["conv_encoder.6.weight"].data.copy_(param.data)
            conv_ox_net.conv_encoder[6].weight.requires_grad = False
        elif name == "conv4.bias":
            conv_ox_net.state_dict()["conv_encoder.6.bias"].data.copy_(param.data)
            conv_ox_net.conv_encoder[6].bias.requires_grad = False

    for name, param in ox_net.state_dict().items():
        if name == "encoder.0.weight":
            conv_ox_net.state_dict()["linear_encoder.0.weight"].data.copy_(param.data)
            conv_ox_net.linear_encoder[0].weight.requires_grad = False
        elif name == "encoder.0.bias":
            conv_ox_net.state_dict()["linear_encoder.0.bias"].data.copy_(param.data)
            conv_ox_net.linear_encoder[0].bias.requires_grad = False
        elif name == "encoder.2.weight":
            conv_ox_net.state_dict()["linear_encoder.2.weight"].data.copy_(param.data)
            conv_ox_net.linear_encoder[2].weight.requires_grad = False
        elif name == "encoder.2.bias":
            conv_ox_net.state_dict()["linear_encoder.2.bias"].data.copy_(param.data)
            conv_ox_net.linear_encoder[2].bias.requires_grad = False
        elif name == "decoder.0.weight":
            conv_ox_net.state_dict()["linear_decoder.0.weight"].data.copy_(param.data)
            conv_ox_net.linear_decoder[0].weight.requires_grad = False
        elif name == "decoder.0.bias":
            conv_ox_net.state_dict()["linear_decoder.0.bias"].data.copy_(param.data)
            conv_ox_net.linear_decoder[0].bias.requires_grad = False
        elif name == "decoder.2.weight":
            conv_ox_net.state_dict()["linear_decoder.2.weight"].data.copy_(param.data)
            conv_ox_net.linear_decoder[2].weight.requires_grad = False
        elif name == "decoder.2.bias":
            conv_ox_net.state_dict()["linear_decoder.2.bias"].data.copy_(param.data)
            conv_ox_net.linear_decoder[2].bias.requires_grad = False

    optimizer = optim.Adam(conv_ox_net.parameters(), lr=1e-4, weight_decay=0)
    target_conv_ox_net = conv_ox_net
    base_image = gather_base_image(bottleneck_data_path)
    train(conv_ox_net, obs_train_data, obs_test_data, optimizer, "./resources/pongD_deconv_obs_model_v1.p", "./data", 32, num_epoch, target_conv_ox_net, cuda=False, grad_clip=10, env=env, low=0, high=0.05, target_test_episodes=1, base_background=None)
    # train(conv_ox_net, obs_train_data, obs_test_data, optimizer, "./resources/pongD_deconv_obs_model_v2.p", "./data", 32, num_epoch, target_conv_ox_net, cuda=False, grad_clip=10, env=env, low=0, high=0.05, target_test_episodes=1, base_background=base_image)