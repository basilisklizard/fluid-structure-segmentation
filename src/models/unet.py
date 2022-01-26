import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, fsize, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch,
                               kernel_size=fsize, stride=stride,
                               padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch,
                               kernel_size=fsize, stride=stride,
                               padding=padding)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class UEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.enc_blocks = nn.ModuleList(
            [Block(ly['input'], ly['output'], ly['f_size'],
                   ly['stride'], ly['padding'])
             for ly in config['encoder']['layers'][1::1]])

    def forward(self, x):
        out = x
        out_list = []
        for block in self.enc_blocks:
            out = self.pooling(out)
            out = block(out)
            out_list.append(out)
        return out_list


class UDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dec_blocks = nn.ModuleList(
            [Block(ly['input'], ly['output'], ly['f_size'],
                   ly['stride'], ly['padding'])
             for ly in config['decoder']['layers']])
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(ly['input'], ly['output'],
                                kernel_size=2, stride=2, padding=0)
             for ly in config['decoder']['layers']])

    def forward(self, x, enc_out):
        out = x
        for i in range(len(self.dec_blocks)):
            out = self.upconvs[i](out)
            out = torch.cat([out, enc_out[i]], dim=1)
            out = self.dec_blocks[i](out)
        return out


class UNet(nn.Module):
    def __init__(self, config, num_classes=1):
        super().__init__()
        self.encoder = UEncoder(config)
        self.decoder = UDecoder(config)

        ly = config['encoder']['layers'][0]
        self.inc = Block(ly['input'], ly['output'], ly['f_size'],
                         ly['stride'], ly['padding'])
        ly = config['decoder']['layers'][-1]
        self.outc = nn.ModuleList(
            [nn.Conv2d(ly['output'], num_classes, 1), nn.Sigmoid()])

        # print model
        print("Encoder ------------------")
        print(self.inc)
        print(self.encoder)
        print("Decoder ------------------")
        print(self.decoder)
        print(self.outc)
        print("")

    def forward(self, x):
        y = self.inc(x)
        y_list = self.encoder(y)
        latent_space = y_list[-1]
        y_list.insert(0, y)
        y = y_list[-1]
        y = self.decoder(y, y_list[:-1][::-1])
        for out_layer in self.outc:
            y = out_layer(y)
        return latent_space, y
