import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
from scipy.sparse import csr_matrix
import utils
from svm import DrebinSVM


class DataProcessor():
    def __init__(self, load_path='./middle_data'):
        self.load_path = load_path
        self.features = None
        self.feature_dict = dict()

        self.good_dataset = None
        self.good_loader = None

        self.mal_dataset = None
        self.mal_loader = None

        self.iter_mal = None

    def load_feature(self):
        self.features = utils.import_from_pkl(self.load_path, 'feature_list.data')

        for i, feature in enumerate(self.features):
            self.feature_dict[feature] = i

        return len(self.features)

    def load_data(self, load_path, batch_size):
        load_abs_path = utils.get_absolute_path(load_path)
        good_sample_path = os.path.join(load_abs_path, 'good_sample')
        mal_sample_path = os.path.join(load_abs_path, 'mal_sample')

        self.good_dataset = DrebinDataset(good_sample_path)
        self.mal_dataset = DrebinDataset(mal_sample_path)

        self.good_loader = DataLoader(self.good_dataset, batch_size=batch_size, shuffle=True)
        self.mal_loader = DataLoader(self.mal_dataset, batch_size=batch_size, shuffle=True)

    def get_random_mal_samples(self):
        try:
            mal_samples = next(self.iter_mal)
        except (StopIteration, TypeError):
            self.iter_mal = iter(self.mal_loader)
            mal_samples = next(self.iter_mal)

        return mal_samples

    def gan2svm(self, x):
        return csr_matrix(x.numpy())

class DrebinDataset(Dataset):
    def __init__(self, load_path):
        self.sample_files = np.array([x.path for x in os.scandir(load_path) if x.name.endswith('.feature')])

    def __getitem__(self, index):
        sample = utils.import_from_pkl(self.sample_files[index])
        return torch.BoolTensor(sample)

    def __len__(self):
        return len(self.sample_files)


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


class WGAN_GP(object):
    def __init__(self, load_path='./middle_data', save_path ='./final_data', model_path='./save_model'):
        self.load_path = load_path
        self.save_path = save_path
        self.model_path = model_path

        # hyper-parameters
        self.gpu = False
        self.model_name = 'GAN'
        self.batch_size = 256
        self.n_critic = 5
        self.max_epoch = 100
        self.lambda_ = 10
        self.lrG = 0.0001
        self.lrD = 0.0001
        self.beta_1 = 0.5
        self.beta_2 = 0.999

        # data loader
        self.DP = DataProcessor(load_path)
        self.feature_size = self.DP.load_feature()

        # gen and dis module
        self.G = Generator(self.feature_size, self.feature_size)
        self.D = Discriminator(self.feature_size, 1)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta_1, self.beta_2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lrD, betas=(self.beta_1, self.beta_2))

        if self.gpu:
            self.G.cuda()
            self.D.cuda()

        # cls module
        self.C = DrebinSVM()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_losses'] = []
        self.train_hist['G_losses'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        # Logger.info("Loading data...")

        self.DP.load_data(self.load_path, self.batch_size)
        # batch_mal = utils.create_batch(mal_samples, self.batch_size)

        # Logger.info("Starting training GAN...")
        self.G.train()
        self.D.train()
        start_time = time.time()
        for epoch in range(self.max_epoch):
            epoch_start_time = time.time()
            G_epoch_loss = 0.
            D_epoch_loss = 0.

            for good_samples in self.DP.good_loader:
                if self.gpu:
                    good_samples.cuda()
                # Generator update
                for param in self.D.parameters():
                    param.requires_grad = False

                self.G_optimizer.zero_grad()

                # Train generator
                # Compute loss with adversarial samples created by G
                mal_samples = self.DP.get_random_mal_samples()
                z = torch.randint(0, 2, (mal_samples.shape[0], self.feature_size), dtype=torch.bool)
                # z = np.random.randint(0, 2, (mal_samples.shape[0], self.feature_size))
                if self.gpu:
                    mal_samples, z = mal_samples.cuda(), z.cuda()

                # z = Variable(torch.Tensor(mal_samples | z))
                z = Variable(mal_samples | z)
                adversarial_samples = self.G(z.float())

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

                    # Generator samples
                    z = torch.randint(0, 2, (mal_samples.shape[0], self.feature_size), dtype=torch.bool)
                    if self.gpu:
                        z = z.cuda()
                    # z = Variable(torch.Tensor(mal_samples | z))
                    z = Variable(mal_samples | z)
                    adversarial_samples = self.G(z).detach()

                    # Gain results from SVM classifier
                    cls_input = torch.cat(adversarial_samples, good_samples)

                    indices = list(range(len(cls_input)))
                    np.random.shuffle(indices)
                    cls_input = self.DP.gan2svm(cls_input[indices])
                    cls_pred = self.C.predict(cls_input)

                    x = cls_input.numpy()[cls_pred == -1] # x as goodware which marks by SVM classifier
                    z = cls_input.numpy()[cls_pred == 1] # z as malware which marks by SVM classifier
                    if self.gpu:
                        x, z = x.cuda(), z.cuda()
                    # Train with good samples
                    D_good = self.D(Variable(torch.Tensor(x)))
                    D_loss_good = torch.mean(D_good)

                    # Train with mal samples
                    D_mal = self.D(Variable(torch.Tensor(z)))
                    D_loss_mal = torch.mean(D_mal)

                    # compute gradient penalty
                    gradient_penalty = self.compute_gradient_penalty(D_good.data, D_mal.data)

                    D_loss = D_loss_mal - D_loss_good + gradient_penalty
                    D_cur_loss += D_loss.item()

                    D_loss.backward()
                    self.D_optimizer.step()

                self.train_hist['D_loss'].append(D_cur_loss / self.n_critic)

            self.train_hist['G_loss'].append(G_epoch_loss)
            self.train_hist['D_loss'].append(D_epoch_loss / self.n_critic)
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            # Logger.info("{} - G_loss: {}\tD_loss: {}".format(epoch, G_epoch_loss, D_epoch_loss / self.n_critic))


        self.train_hist['total_time'].append(time.time() - start_time)
        # Logger.info('Average one epoch time: {:.2f}, total {:d} epochs time: {:.2f}' \
        #     .format(np.mean(self.train_hist['per_epoch_time']), self.max_epoch, self.train_hist['total_time'][0]))
        # Logger.info("GAN training Done.")

        # save model
        # Logger.info("Saving training model...")
        self.save()

    def save(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        torch.save(self.G.state_dict(), os.path.join(self.model_path, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(self.model_path, self.model_name + '_D.pkl'))

    def load(self):
        self.G.load_state_dict(torch.load(os.path.join(self.model_path, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(self.model_path, self.model_name + '_D.pkl')))

    def compute_gradient_penalty(self, D_good, D_mal):
        alpha = torch.Tensor(np.random.random((D_good.shape[0], 1)))
        if self.gpu:
            alpha = alpha.cuda()

        interpolation = alpha * D_good + (1 - alpha) * D_mal
        interpolation.requires_grad = True

        D_inter = self.D(interpolation)
        adv = Variable(torch.Tensor(D_good.shape[0], 1).fill_(1), requires_grad=False)

        if self.gpu:
            gradients = autograd.grad(
                outputs=D_inter,
                inputs=interpolation,
                grad_outputs=adv.cuda(),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        else:
            gradients = autograd.grad(
                outputs=D_inter,
                inputs=interpolation,
                grad_outputs=adv,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def generate(self):
        # load model
        # forward generator
        # return output features
        pass


if __name__ == '__main__':
    # utils.preprocess_data()
    gan = WGAN_GP()
    gan.train()

    print(1)