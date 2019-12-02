import torch

from net import Net


def train(params, *, gradient_type, implementation='old'):

    torch.manual_seed(params['seed'])

    # teacher network
    model_target = Net()

    # student network
    model = Net()

    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])

    history_loss = torch.empty(params['training_batches'])
    for step in range(params['training_batches']):

        x = torch.empty(params['batch_size'], 3).uniform_(-1.5, 1.5)

        # generate target output
        with torch.no_grad():
            y_target = model_target(x)


        # perform weight update
        if gradient_type == 'euclidean':
            y = model(x)
            loss = torch.mean((y - y_target) ** 2)
            model.zero_grad()
            loss.backward()

        elif gradient_type == 'quasi-diagonal-natural':
            loss = model.compute_natural_grad(x, y_target)

        else:
            raise NotImplementedError()

        optimizer.step()

        history_loss[step] = loss.item()

    return history_loss
