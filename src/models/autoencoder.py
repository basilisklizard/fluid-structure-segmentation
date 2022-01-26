import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential()
        for i in range(len(config['encoder']['layers'])):
            ly = config['encoder']['layers'][i]
            self.encoder.add_module(
                'e_conv'+str(i),
                nn.Conv2d(ly['input'], ly['output'], ly['f_size'],
                          stride=ly['stride'], padding=ly['padding'])
            )
            self.encoder.add_module(
                'e_pool'+str(i),
                nn.MaxPool2d(2, stride=2, ceil_mode=True)
            )
            self.encoder.add_module(
                'e_relu'+str(i),
                nn.ReLU()
            )
        self.decoder = nn.Sequential()
        for i in range(len(config['decoder']['layers'])):
            ly = config['decoder']['layers'][i]
            self.decoder.add_module(
                'd_upsample'+str(i),
                nn.Upsample(scale_factor=2)
            )
            self.decoder.add_module(
                'd_conv'+str(i),
                nn.Conv2d(ly['input'], ly['output'], ly['f_size'],
                          stride=ly['stride'], padding=ly['padding'])
            )
            if i < len(config['decoder']['layers'])-1:
                self.decoder.add_module(
                    'd_relu'+str(i),
                    nn.ReLU()
                )
            else:
                self.decoder.add_module(
                    'd_sigmoid'+str(i),
                    nn.Sigmoid()
                )
        # print model
        print("Encoder ------------------")
        print(self.encoder)
        print("Decoder ------------------")
        print(self.decoder)
        print("")

    def forward(self, x):
        latent_space = self.encoder(x)
        output = self.decoder(latent_space)
        return latent_space, output
