import torch
import argparse
import train
import module
import preprocess


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2, help='batch大小(默认为2)')
parser.add_argument('--epochs', type=int, default=5000, help='epochs大小(默认为5000)')
parser.add_argument('--n_step', type=int, default=2, help='输入的文本单词数(默认为2)')
parser.add_argument('--n_hidden', type=int, default=5, help='隐藏层的维度')
parser.add_argument('--n_class', type=int, default=6, help='分类数(默认为6)')
args = parser.parse_args()
args.device = torch.device('cpu')


sentences = [ "i like dog", "i love coffee", "i hate milk"]
word_list = " ".join(sentences).split()
vocab = list(set(word_list))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
args.n_class = len(vocab)

loader = preprocess.make_data(sentences, word2idx, args)
model = module.TextRNN(args)
train.train(args, model, loader)


