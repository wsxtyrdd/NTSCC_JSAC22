import torch.nn as nn
import numpy as np
import os
import torch


class Channel(nn.Module):
    def __init__(self, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = config.channel['type']
        self.chan_param = config.channel['chan_param']
        self.device = config.device
        if config.logger:
            config.logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                config.channel['type'], config.channel['chan_param']))

    def gaussian_noise_layer(self, input_layer, std):
        device = input_layer.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def forward(self, input, avg_pwr, power=1):
        channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        channel_in = channel_in[::2] + channel_in[1::2] * 1j
        channel_output = self.channel_forward(channel_in)
        channel_rx = torch.zeros_like(channel_tx.reshape(-1))
        channel_rx[::2] = torch.real(channel_output)
        channel_rx[1::2] = torch.imag(channel_output)
        channel_rx = channel_rx.reshape(input_shape)
        return channel_rx * torch.sqrt(avg_pwr * 2)

    def channel_forward(self, channel_in):
        if self.chan_type == 0 or self.chan_type == 'none':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (self.chan_param / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma)
            return chan_output