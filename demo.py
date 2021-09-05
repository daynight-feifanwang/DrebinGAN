import os.path

from drebin_svm import DrebinSVM
from drebin_gan import DrebinGAN
import numpy as np
import utils
from scipy.sparse import csr_matrix, vstack

def generate_adversarial_sample(load_path='./test_data', library_scale=100):
    # init module
    # test_data dir contains {mal_samples, benign_features, total_features}
    generator = DrebinGAN(load_path=load_path, save_path=load_path)
    classifier = DrebinSVM(load_path=load_path, save_path=load_path)

    # load model
    generator.load('DrebinGAN_G_3499.pkl', 'DrebinGAN_D_3499.pkl')
    classifier.load()

    # load feature map
    benign2total_map, feature_size = utils.benign2total(load_path)
    benign2total_map = np.array(benign2total_map)

    # generate fake masks, fake mask type: {n_darray: {num, benign_feat_size}}
    masks = generator.generate(library_scale)

    # map masks to total features
    features = np.nonzero(masks) # benign feature index with mark 1
    row = features[0]
    col = benign2total_map[features[1]]
    data = np.ones(col.shape[0])

    benign_samples = csr_matrix((data, (row, col)), shape=(masks.shape[0], feature_size))

    # load mal samples, type: {0,0,1,0, ... , 1,0}
    mal_samples = np.array([utils.import_from_pkl(x.path) for x in os.scandir(load_path) if x.name.endswith('.feature')])

    adversarial_samples = []

    for mal_sample in mal_samples:
        mal_sample = csr_matrix(mal_sample)
        mal_sample = vstack([mal_sample] * library_scale)

        adversarial_sample = utils.sparse_maximum(benign_samples, mal_sample)

        result = classifier.predict(adversarial_sample)

        result = np.where(result == -1) # filter out the good samples classified
        if result[0].shape[0] != 0:
            adversarial_sample = adversarial_sample[result]
            mal_sample = mal_sample[result]

            diff = ((adversarial_sample - mal_sample) == 1).sum(1)
            best_sample = adversarial_sample[diff.argmin()]

            adversarial_samples.append([best_sample, diff.min()])
        else:
            adversarial_samples.append(None)

    return adversarial_samples


if __name__ == '__main__':
    ad = generate_adversarial_sample(library_scale=1000)
    print(ad)



