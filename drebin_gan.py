import os
import time
import numpy as np
import torch
import logging

from scipy.sparse import csr_matrix, vstack
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder
import utils
from torch.utils.tensorboard import SummaryWriter
from model import Generator, Discriminator
from drebin_svm import DrebinSVM

logging.basicConfig(level=logging.INFO, format="'%(asctime)s - %(name)s: %(levelname)s: %(message)s'")
logger = logging.getLogger("DrebinGAN.STDOUT")
logger.setLevel("INFO")

torch.manual_seed(10)
use_cuda = torch.cuda.is_available()


class DataProcessor:
    def __init__(self, load_path='./middle_data'):
        self.load_path = load_path

        self.dataset = None
        self.loader = None
        self.iterator = None

    def load_data(self, load_path, file_name, batch_size):
        load_abs_path = utils.get_absolute_path(load_path)
        sample_path = os.path.join(load_abs_path, file_name)

        self.dataset = DrebinDataset(sample_path)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def next(self):
        try:
            samples = next(self.iterator)
        except (StopIteration, TypeError):
            self.iterator = iter(self.loader)
            samples = next(self.iterator)
        return samples


class DrebinDataset(Dataset):
    def __init__(self, load_path):
        self.sample_files = np.array([x.path for x in os.scandir(load_path) if x.name.endswith('.feature')])

    def __getitem__(self, index):
        sample = utils.import_from_pkl(self.sample_files[index])
        return sample

    def __len__(self):
        return len(self.sample_files)


class DrebinGAN(object):
    def __init__(self, load_path='./final_data', save_path='./final_data', model_path='./save_model',
                 classifier_model=DrebinSVM):
        self.load_path = load_path
        self.save_path = save_path
        self.model_path = model_path

        # hyper-parameters
        self.model_name = 'DrebinGAN'
        self.batch_size = 128
        self.dim = 128 # need to adjust
        self.n_dim = 128
        self.n_critic = 5
        self.max_epoch = 100000
        self.lambda_ = 10
        self.lrG = 0.0001
        self.lrD = 0.0001
        self.beta_1 = 0.5
        self.beta_2 = 0.999

        # data loader
        self.DP = None
        self.benign_feature_size = self.load_feature()
        logger.info("Feature size: {}".format(self.benign_feature_size))

        # gen and dis module
        self.G = Generator(self.dim, self.n_dim, self.benign_feature_size, 2)
        self.D = Discriminator(self.dim, self.benign_feature_size, 2)
        self.G_optim = optim.Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta_1, self.beta_2))
        self.D_optim = optim.Adam(self.D.parameters(), lr=self.lrD, betas=(self.beta_1, self.beta_2))

        # classifier model
        self.classifier = classifier_model
        self.map = None

        if use_cuda:
            self.G.cuda()
            self.D.cuda()

    def load_feature(self):
        features = utils.import_from_pkl(self.load_path, 'benign_feature_list.data')
        return len(features)

    def train(self):
        # init
        one_hot = OneHotEncoder()
        one_hot.fit(np.array([[0], [1]]))
        writer = SummaryWriter('logs')

        one = torch.tensor(1, dtype=torch.float)
        minus_one = one * -1

        one = one.cuda() if use_cuda else one
        minus_one = minus_one.cuda() if use_cuda else minus_one

        logger.info("Loading data...")
        self.DP = DataProcessor(self.load_path)
        self.DP.load_data(self.load_path, 'good_sample', self.batch_size)

        logger.info("Starting training GAN...")
        self.G.train()
        self.D.train()
        start_time = time.time()
        for epoch in range(self.max_epoch):
            epoch_start_time = time.time()

            # Update Discriminator
            for parameter in self.D.parameters():  # reset requires_grad every time
                parameter.requires_grad = True     # as they are blocked during Generator updating.
            for i_critic in range(self.n_critic):
                self.D.zero_grad()

                # get good samples and reshape
                real_data = self.DP.next()  # real_data = {ndarray: (batch_size, feat_size)}
                real_data = one_hot.transform(real_data.reshape(-1, 1)).toarray().reshape(-1, self.benign_feature_size, 2)
                real_data = torch.Tensor(real_data)
                real_data = real_data.cuda() if use_cuda else real_data
                real_data_v = Variable(real_data)

                # train discriminator with real data
                D_real = self.D(real_data_v)
                D_real = D_real.mean()
                D_real.backward(minus_one)

                # train discriminator with fake data generated by generator
                noise = torch.randn(real_data_v.shape[0], self.n_dim)
                noise = noise.cuda() if use_cuda else noise
                noise = Variable(noise, volatile=True)

                fake_data = self.G(noise)
                fake_data_v = Variable(fake_data.data)

                G_fake = fake_data_v
                D_fake = self.D(G_fake)
                D_fake = D_fake.mean()
                D_fake.backward(one)

                # update with grdient penalty
                gp = self.compute_gradient_penalty(real_data_v.data, fake_data_v.data)
                gp.backward()


                D_loss = D_fake - D_real + gp
                D_wasserstein = D_real - D_fake
                self.D_optim.step()

            # Update Generator
            for parameter in self.D.parameters():
                parameter.requires_grad = False
            self.G.zero_grad()

            noise = torch.randn(self.batch_size, self.n_dim)
            noise = noise.cuda() if use_cuda else noise
            noise = Variable(noise)

            G_fake = self.G(noise)
            D_fake = self.D(G_fake)
            D_fake = D_fake.mean()
            D_fake.backward(minus_one)

            G_loss = -D_fake
            self.G_optim.step()

            # print log
            # Logger.info("{} - G_loss: {}\tD_loss: {}".format(epoch, G_epoch_loss, D_epoch_loss / self.n_critic))
            print('[{}] - D_loss: {}, G_loss: {}, W_loss: {} ... epoch_time: '.format(epoch,
                                                                                      D_loss.item(),
                                                                                      G_loss.item(),
                                                                                      D_wasserstein.item())
                  + utils.consume_time(time.time() - epoch_start_time))
            # plot log
            writer.add_scalar('Discriminator Loss', D_loss.item(), epoch)
            writer.add_scalar('Generator Loss', G_loss.item(), epoch)
            writer.add_scalar('Wasserstein Distance', D_wasserstein.item(), epoch)

            # evaluate and save model
            if epoch % 500 == 499:
                # generate samples
                self.evaluate(writer, epoch=epoch)
            if epoch % 500 == 499:
                # save model
                # print('[{}] - Saving model'.format(epoch))
                self.save(epoch)

        writer.close()

    def evaluate(self, writer, load_path='./final_data', num_n=100, base_n=100, epoch=None):
        # try multiprocess
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool()
        # load classifier
        if type(self.classifier) == type:
            self.classifier = DrebinSVM(load_path=load_path)
            self.classifier.load()

        if self.map is None:
            self.map, self.total_feature_size = utils.benign2total(load_path)
            self.map = np.array(self.map)

        fake_path = os.path.join(utils.get_absolute_path(load_path), 'fake_good_sample')
        if not os.path.exists(fake_path):
            os.makedirs(fake_path)

        # generate fake benign samples
        print('[{}] - Creating noises'.format(epoch), end='\r')
        masks = []
        for i in range(num_n // base_n):
            masks.append(self.generate(base_n))
        masks = np.array(masks).reshape(num_n, self.benign_feature_size)
        np.save(os.path.join(fake_path, 'fake_sample_{}.npy'.format(epoch)), masks)
        print('[{}] - Creating noises ... Done'.format(epoch), end='\r')

        features = np.nonzero(masks)  # benign feature index with mark 1
        row = features[0]
        col = self.map[features[1]]
        data = np.ones(col.shape[0])

        mal_path = os.path.join(utils.get_absolute_path(load_path), 'mal_sample')
        benign_samples = csr_matrix((data, (row, col)), shape=(masks.shape[0], self.total_feature_size))
        mal_samples = [x.path for x in os.scandir(mal_path) if x.name.endswith('.feature')]

        for i, mal_sample in enumerate(mal_samples):
            mal_samples[i] = (i, mal_sample)

        SMR = []
        AMC = []
        start_time = time.time()
        '''
        for i, mal_sample in enumerate(mal_samples):
            if i % 300 == 299:
                print('[{}] - Evaluated {}/{} samples, consuming time:'.format(epoch, i + 1, len(mal_samples))
                      + utils.consume_time(time.time() - start_time), end='\r')
            mal_sample = csr_matrix(utils.import_from_pkl(mal_sample))
            mal_sample = vstack([mal_sample] * num_n)

            adversarial_sample = utils.sparse_maximum(benign_samples, mal_sample)

            result = self.classifier.predict(adversarial_sample)
            escape = np.where(result == -1)

            if escape[0].shape[0] != 0:
                evaded_sample = adversarial_sample[escape]
                mal_sample = mal_sample[escape]

                diff = ((evaded_sample - mal_sample) == 1).sum(1)
                mean_diff = diff.mean()

                escape_rate.append(escape[0].shape[0] / num_n)
                diff_cost.append(mean_diff)
            else:
                escape_rate.append(0)
                diff_cost.append(-1)
        '''

        def process(mal_sample):
            i, mal_sample = mal_sample
            if i % 720 == 719:
                print('[{}] - Evaluated {}/{} samples, consuming time:'.format(epoch, i + 1, len(mal_samples))
                      + utils.consume_time(time.time() - start_time))
            mal_sample = csr_matrix(utils.import_from_pkl(mal_sample))
            mal_sample = vstack([mal_sample] * num_n)
            adversarial_sample = utils.sparse_maximum(benign_samples, mal_sample)
            result = self.classifier.predict(adversarial_sample)
            escape = np.where(result == -1)
            if escape[0].shape[0] != 0:
                evaded_sample = adversarial_sample[escape]
                mal_sample = mal_sample[escape]

                diff = ((evaded_sample - mal_sample) == 1).sum(1)
                mean_diff = diff.min()

                return escape[0].shape[0] / num_n, mean_diff
            else:
                return 0, -1

        results = pool.map(process, mal_samples)
        for result in results:
            SMR.append(result[0])
            AMC.append(result[1])
        pool.close()
        pool.join()

        # save the result
        utils.export_to_pkl('./report', 'report_escape_rate_{}.pkl'.format(epoch), SMR)
        utils.export_to_pkl('./report', 'report_diff_cost_{}.pkl'.format(epoch), AMC)

        # plot the result
        SMR = np.mean(np.array(SMR))
        SER = np.count_nonzero(SMR)
        AMC = np.array(AMC)
        try:
            AMC = round(np.mean(AMC[AMC != -1]))
        except Exception:
            AMC = 4096

        print('[{}] - SMR: {:.4f}, ESR: {:.4f}, AMC: {:d}  consuming time:'.format(epoch, SMR, SER, AMC)
              + utils.consume_time(time.time() - start_time))
        writer.add_scalar('Successfully Modified Ratio', SMR, epoch)
        writer.add_scalar('Successfully Escaped Ratio', SER, epoch)
        writer.add_scalar('Average Modification Cost', AMC, epoch)
        return masks

    def save(self, epoch):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        torch.save(self.G.state_dict(), os.path.join(self.model_path, self.model_name + '_G_{:d}.pkl'.format(epoch)))
        torch.save(self.D.state_dict(), os.path.join(self.model_path, self.model_name + '_D_{:d}.pkl'.format(epoch)))

    def load(self, G_file_name, D_file_name):
        self.G.load_state_dict(torch.load(os.path.join(self.model_path, G_file_name)))
        self.D.load_state_dict(torch.load(os.path.join(self.model_path, D_file_name)))

    def compute_gradient_penalty(self, D_real, D_fake):
        alpha = torch.rand(D_real.shape[0], 1, 1).expand(D_real.size())
        alpha = alpha.cuda() if use_cuda else alpha

        interpolates = alpha * D_real + (1 - alpha) * D_fake
        interpolates = interpolates.cuda() if use_cuda else alpha
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        D_inter = self.D(interpolates)

        ones = torch.ones(D_inter.size()).cuda() if use_cuda else torch.ones(D_inter.size())

        gradients = autograd.grad(outputs=D_inter,
                                  inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_
        return gradient_penalty

    def generate(self, num=1, saving=False, save_path=None):
        noise = torch.randn(num, self.n_dim)
        noise = noise.cuda() if use_cuda else noise
        with torch.no_grad():
            samples = self.G(noise)
            samples = samples.view(-1, self.benign_feature_size, 2)
            _, samples = torch.max(samples, 2)
            samples = samples.cpu().data.numpy()

        if saving and save_path:
            utils.export_to_pkl(save_path, content=samples)
        return samples


if __name__ == '__main__':
    utils.process_data()
    gan = DrebinGAN(classifier_model=DrebinSVM)
    gan.train()
