import torch
from torch import nn
import einops
import torch.nn.functional as F


# trainable embedding with slim mlp
class DoraNet(nn.Module):
    def __init__(self,embed_dim=8, embed_width=32, test=False):
        # origin embed_dim=8
        super(DoraNet, self).__init__()
        self.embed_dim = embed_dim
        self.embed_width = embed_width
        # self.embed1 = nn.Parameter(torch.zeros(self.embed_dim, self.embed_width, self.embed_width))
        # self.embed2 = nn.Parameter(torch.zeros(self.embed_dim, self.embed_width, self.embed_width))
        # self.embed3 = nn.Parameter(torch.zeros(self.embed_dim, self.embed_width, self.embed_width))
        # self.embed4 = nn.Parameter(torch.zeros(self.embed_dim, self.embed_width, self.embed_width))

        self.embeds = list()
        for i in range(4):
            # self.embeds.append(nn.Parameter(torch.randn(self.embed_dim, self.embed_width, self.embed_width)))

            self.embeds.append(nn.Parameter(torch.zeros(self.embed_dim, self.embed_width, self.embed_width)))

        self.embeds = nn.ParameterList(self.embeds)
        # self.decoder = nn.Sequential(
        #     nn.Linear(self.embed_dim + 2, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # )

        self.decoders = list()
        for i in range(4):
            self.decoders.append(
                nn.Sequential(
                    nn.Linear(self.embed_dim + 2, 32),
                    # nn.LekyReLU(),
                    nn.LayerNorm(32),
                    nn.LeakyReLU(),
                    nn.Linear(32, 32),
                    nn.LayerNorm(32),
                    nn.LeakyReLU(),
                    nn.Linear(32, 32),
                    nn.LayerNorm(32),
                    nn.LeakyReLU(),
                    nn.Linear(32, 32),
                    nn.LayerNorm(32),
                    nn.LeakyReLU(),
                    nn.Linear(32, 32),
                    nn.LayerNorm(32),
                    nn.LeakyReLU(),
                    nn.Linear(32, 32),
                    nn.LayerNorm(32),
                    nn.LeakyReLU(),

                    # add by jiaxi
                    nn.Linear(32, 32),
                    nn.LayerNorm(32),
                    nn.LeakyReLU(),
                    nn.Linear(32, 32),
                    nn.LayerNorm(32),
                    nn.LeakyReLU(),

                    nn.Linear(32, 32),
                    nn.LeakyReLU(),
                    nn.Linear(32, 1)
                )
            )

        self.decoders = nn.ModuleList(self.decoders)

        # intial
        for i in range(4):
            torch.nn.init.kaiming_normal_(self.embeds[i])

        self.step = (355. - (-21)) / (self.embed_width - 1)
                
    def forward(self, pos):
        # normalize pos
        pos = (pos - (-21.))/(355 - (-21.))  # [0, 1]
        pos = pos * 2 - 1  # [-1, 1]

        # pos_res = pos - (pos / self.step).int().float()
        # grid_sample implement
        pathloss = list()
        for i in range(4):
            embedding = einops.repeat(self.embeds[i], 'c h w -> b c h w', b=1)
            grids = einops.repeat(pos, 'n x -> b n m x', b=1, m=1)
            feature = F.grid_sample(embedding, grids, mode='bilinear', align_corners=True)
            # feature in shape [1, c, n, 1]
            # print(feature.shape)
            assert feature.shape[0] == 1 and feature.shape[-1] == 1
            feature = feature.squeeze(0).squeeze(2)
            feature = einops.rearrange(feature, 'c n -> n c')
            feature = torch.cat([feature, pos], dim=-1)
            y = self.decoders[i](feature)
            pathloss.append(y)
            # print(y.shape)
        pathloss = torch.cat(pathloss, dim=-1)

        return pathloss

