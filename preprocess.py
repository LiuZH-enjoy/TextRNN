import torch.utils.data as Data
import torch
import numpy as np


def make_data(sentences, word2idx, args):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word2idx[n] for n in word[:-1]]
        target = word2idx[word[-1]]

        input_batch.append(np.eye(args.n_class)[input])
        target_batch.append(target)
        input_batch, target_batch = torch.Tensor(input_batch), torch.LongTensor(target_batch)
        dataset = Data.TensorDataset(input_batch, target_batch)
        loader = Data.DataLoader(dataset, args.batch_size, True)
        return loader
