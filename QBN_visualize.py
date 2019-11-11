import pickle
import torch
import torch.nn as nn
from env_wrapper import atari_wrapper
from visTorch import visboard
from torchvision.utils import make_grid
from PIL import Image
from additional_functions import TernaryTanh

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
        # linear_encoder_input = conv_encoded.view(-1, self.qbn_input_size)
        linear_encoder_input = torch.reshape(conv_encoded, (x.shape[0], self.qbn_input_size))
        linear_encoded = self.linear_encoder(linear_encoder_input)
        return linear_encoded

    def decode(self, x):
        linear_decoded = self.linear_decoder(x)
        # conv_decoder_input = torch.reshape(linear_decoded, (1, 8, 5, 5))
        conv_decoder_input = torch.reshape(linear_decoded, (x.shape[0], 8, 5, 5))
        conv_decoded = self.conv_decoder(conv_decoder_input)
        return conv_decoded

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.gru_units)


def process_to_img(x):
    # x = x.view(8, 5, 5)
    x = torch.Tensor(x)
    x = torch.reshape(x, (80, 80))
    grid = make_grid(x)
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # ndarr = Variable(torch.Tensor(x)).unsqueeze(0).data.numpy()
    im = Image.fromarray(ndarr)
    # im = Image.fromarray((ndarr * 255).astype(np.uint8)[0], mode="L")
    return im


episodes = 2
gru_size = 32
bhx_size = 64
ox_size = 100
input_c_features = 8 * 5 * 5
eps = (0, 0)
# hx_ae_model = HxQBNet(gru_size, bhx_size)
ox_ae_best_path = "./resources/pongD_conv_obs_model.p"

env_name = "PongDeterministic-v4"
env = atari_wrapper(env_name)
obs = env.reset()

ox_ae_model = ConvObsQBNet(len(obs), ox_size)

# initialize visualization app
_, _, obs_data, _ = pickle.loads(open("./resources/pongD_bottleneck_data.p", "rb").read())

vis_board = visboard()
vis_board.add_ae(ox_ae_model,
                 obs_data,
                 latent_options={'n': ox_ae_model.latent_size, 'min': -1, 'max': 1, 'step': 1},
                 model_paths={'Best': ox_ae_best_path},
                 pre_process=process_to_img)
vis_board.run_server('127.0.0.1', '8051')