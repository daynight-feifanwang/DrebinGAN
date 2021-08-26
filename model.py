import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 5, padding=2),  # nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 5, padding=2),  # nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3 * output)

class Generator(nn.Module):
    def __init__(self, dim, n_dim, feat_size, val_size):
        super(Generator, self).__init__()
        self.dim = dim
        self.feat_size = feat_size

        self.fc1 = nn.Linear(n_dim, dim * feat_size)
        self.block = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        )
        self.conv1 = nn.Conv1d(dim, val_size, 1)
        self.softmax = nn.Softmax()

    def forward(self, noise):
        batch_size = noise.size(0)
        output = self.fc1(noise)
        output = output.view(-1, self.dim, self.feat_size) # BATCH_SIZE, DIM, FEAT_SIZE
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(batch_size * self.feat_size, -1)
        output = self.softmax(output)
        # (BATCH_SIZE, FEAT_SIZE, VAL_SIZE)
        return output.view(shape)


class Discriminator(nn.Module):
    def __init__(self, dim, feat_size, val_size):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.feat_size = feat_size

        self.block = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        )
        self.conv1d = nn.Conv1d(val_size, dim, 1)
        self.linear = nn.Linear(feat_size * dim, 1)

    def forward(self, input):
        output = input.transpose(1, 2)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, self.feat_size * self.dim)
        output = self.linear(output)
        return output