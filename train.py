import torch
import torch.nn as nn
import torch.optim as optim


def train(args, model, loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5000):
        for x, y in loader:
            # hidden : [num_layers * num_directions, batch, hidden_size]
            hidden = torch.zeros(1, x.shape[0], args.n_hidden)
            # x : [batch_size, n_step, n_class]
            pred = model(hidden, x)

            # pred : [batch_size, n_class], y : [batch_size] (LongTensor, not one-hot)
            loss = criterion(pred, y)
            if (epoch + 1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()