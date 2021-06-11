import torch.nn as nn


class TextRNN(nn.Module):
    def __init__(self, args):
        super(TextRNN, self).__init__()
        self.rnn = nn.RNN(input_size=args.n_class, hidden_size=args.n_hidden)
        self.fc = nn.Linear(args.n_hidden, args.n_class)

    def forward(self, hidden, X):
        # X : [batch_size, n_step, n_class]
        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]
        out, hidden = self.rnn(X, hidden)
        # out ：[n_step, batch_size, num_directions(=1)*n_hidden]
        # hidden : [num_larers(=1)*num_directions(=1), batch_size, n_hidden]
        out = out[-1] # [batch_size, num_directions(=1)*n_hidden]
        model = self.fc(out)
        return model
