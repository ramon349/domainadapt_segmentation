# Nicola Dinsdale 2020
# Model for unlearning domain for segmentation task with more unlearning points
########################################################################################################################
# Import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
########################################################################################################################

from collections import OrderedDict
import pdb 
import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels=1, init_features=4):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._half_block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._half_block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._half_block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._half_block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._half_block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._half_block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._half_block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._half_block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._half_block(features * 2, features, name="dec1")

    def forward(self, x,slide_window_compress=False): 
        if (not self.training )and slide_window_compress: 
            #this is only used in inference when using 
            x = x.squeeze(-1) 

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1) 
        if (not self.training )and slide_window_compress: 
            dec1 = dec1.unsqueeze(-1)
        return  dec1

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )

    @staticmethod
    def _half_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )


class domain_predictor(nn.Module):
    def __init__(self, n_domains=2, init_features=4):
        super(domain_predictor, self).__init__()
        self.n_domains = n_domains
        features = init_features

        self.decoder1 = domain_predictor._half_block(features, features, name="conv1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder2 = domain_predictor._half_block(features, features, name="conv2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder3 = domain_predictor._half_block(features, features, name="conv3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder4 = domain_predictor._half_block(features, features, name="conv3")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder5 = domain_predictor._projector_block(features, 1, name="projectorblock")
        # Projector block to reduce features
        self.lin1 = nn.Linear(512, 96)
        self.relu1 = nn.ReLU(True)


        # Also take them from the bottom of the u
        self.decoder5 = domain_predictor._projector_block(features * 16, 1, name='projectorblock2')
        self.lin2 = nn.Linear(100, 96)
        self.relu2 = nn.ReLU(True)

        self.domain = nn.Sequential()
        self.domain.add_module('r_relu1', nn.ReLU(True))
        self.domain.add_module('d_fc2', nn.Linear(96*2, 32))
        self.domain.add_module('d_relu2', nn.ReLU(True))
        self.domain.add_module('r_dropout', nn.Dropout3d(p=0.2))
        self.domain.add_module('d_fc3', nn.Linear(32, n_domains))
        self.domain.add_module('d_pred', nn.Softmax(dim=1))

    def forward(self, x):
        [x1, x2] = x
        dec1 = self.decoder1(x1)
        dec2 = self.decoder2(self.pool1(dec1))
        dec3 = self.decoder3(self.pool2(dec2))
        dec4  = self.decoder4(self.pool3(dec3)) 
        dec4 = torch.flatten(dec4,1,-1)
        lin1 = self.relu1(self.lin1(dec4))

        dec5 = self.decoder5(x2)
        dec5 = torch.flatten(dec5,1,-1) 
        lin2 = self.relu2(self.lin2(dec5))
        linear = torch.cat((lin1, lin2), dim=1)

        domain_pred = self.domain(linear)
        return domain_pred

    @staticmethod
    def _projector_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )

    @staticmethod
    def _half_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )

class RamenDinsdale2D(UNet): 
    def __init__(self, in_channels=1, init_features=2):
        super().__init__(in_channels, init_features) 
        self.discrim = domain_predictor(n_domains=2,init_features=2)
    def forward(self,x): 
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        phase_preds = self.discrim([dec1,bottleneck])
        return [dec1, phase_preds]

