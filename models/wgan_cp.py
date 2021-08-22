import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import MultiLabelBinarizer
import proj1.utils as utils
import proj1.feature_extract as feature_extract
from svm import DrebinSVM



def transform(samples):
    mlb = MultiLabelBinarizer()
    x = mlb.fit_transform(samples)
    return good_samples, mal_samples


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self, x):
        x = self.layer(x)
        return torch.clamp(x, 0, 1)


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim * 2, input_dim // 2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.layer(x)


class WGAN_CP(object):
    def __init__(self, args):
        self.model_name = 'gan_model_'
        self.data = args.data_dir
        self.save_dir = args.save_dir
        self.feature_dim = utils.get_feature_dim(args.feature_dir)
        self.D_output_dim = 1
        self.batch_size = 256
        self.n_critic = 5
        self.max_epoch = 100
        self.clamp = 0.01
        self.lrG = 0.0001
        self.lrD = 0.0001

        self.G = Generator(self.feature_dim, self.feature_dim)
        self.D = Discriminator(self.feature_dim, self.D_output_dim)
        self.G_optimizer = optim.RMSprop(self.G.parameters(), lr=self.lrG)
        self.D_optimizer = optim.RMSprop(self.D.parameters(), lr=self.lrD)
        self.C = DrebinSVM().load()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_losses'] = []
        self.train_hist['G_losses'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        # Logger.info("Loading data...")
        samples, _, features = feature_extract.LoadData(self.data)
        good_samples, mal_samples = transform(samples, features)
        # Logger.info("Starting training GAN...")
        self.G.train()
        self.D.train()
        start_time = time.time()
        for epoch in range(self.max_epoch):
            epoch_start_time = time.time()
            batch_good = utils.create_batch(good_samples, self.batch_size)
            G_epoch_loss = 0.
            D_epoch_loss = 0.

            for bg in batch_good:
                x = torch.Tensor(bg)
                # Generator update
                for param in self.D.parameters():
                    param.requires_grad = False

                self.G_optimizer.zero_grad()

                # Train generator
                # Compute loss with adversarial samples created by G
                mal_rand = mal_samples[np.random.randint(0, len(mal_samples), self.batch_size)]
                z = np.random.randint(0, 2, (self.batch_size, self.feature_dim))
                z = Variable(torch.Tensor(mal_rand | z))
                adversarial_samples = self.G(z)

                D_fake = self.D(adversarial_samples)
                G_loss = -torch.mean(D_fake)


                G_loss.backward()
                self.G_optimizer.step()

                # Train discriminator
                for param in self.D.parameters():
                    param.requires_grad = True
                # Train discriminator more iterations than generator
                D_cur_loss = 0.
                for i_critic in range(self.n_critic):
                    self.D_optimizer.zero_grad()
                    # clipping D
                    for p in self.D.parameters():
                        p.data.clamp_(-self.clamp, self.clamp)

                    # Generator samples
                    z = np.random.randint(0, 2, (self.batch_size, self.feature_dim))
                    z = Variable(torch.Tensor(mal_rand | z))
                    adversarial_samples = self.G(z).detach()

                    # Gain results from SVM classifier
                    cls_input = torch.cat(adversarial_samples, batch_good)

                    indices = list(range(len(cls_input)))
                    np.random.shuffle(indices)
                    cls_input = Variable(torch.Tensor(cls_input[indices]))
                    cls_pred = self.C.predict(cls_input)

                    x = cls_input.numpy()[cls_pred == -1] # x as goodware which marks by SVM classifier
                    z = cls_input.numpy()[cls_pred == 1] # z as malware which marks by SVM classifier
                    # Train with good samples
                    D_good = self.D(Variable(torch.Tensor(x)))
                    D_loss_good = torch.mean(D_good)

                    # Train with mal samples
                    D_mal = self.D(Variable(torch.Tensor(z)))
                    D_loss_mal = torch.mean(D_mal)

                    D_loss = D_loss_mal - D_loss_good
                    D_cur_loss += D_loss.item()

                    D_loss.backward()
                    self.D_optimizer.step()

                self.train_hist['D_loss'].append(D_cur_loss / self.n_critic)

            self.train_hist['G_loss'].append(G_epoch_loss)
            self.train_hist['D_loss'].append(D_epoch_loss / self.n_critic)
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad:
                self.visualize(epoch + 1)

            # Logger.info("{} - G_loss: {}\tD_loss: {}".format(epoch, G_epoch_loss, D_epoch_loss / self.n_critic))


        self.train_hist['total_time'].append(time.time() - start_time)
        # Logger.info('Average one epoch time: {:.2f}, total {:d} epochs time: {:.2f}' \
        #     .format(np.mean(self.train_hist['per_epoch_time']), self.max_epoch, self.train_hist['total_time'][0]))
        # Logger.info("GAN training Done.")

        # save model
        # Logger.info("Saving training model...")
        self.save()

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        torch.save(self.G.state_dict(), os.path.join(self.save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(self.save_dir, self.model_name + '_D.pkl'))

    def load(self):
        self.G.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '_D.pkl')))

    def generate(self):
        pass
        # load model
        # forward generator
        # return output features

    def visualize(self, epoch):
        pass