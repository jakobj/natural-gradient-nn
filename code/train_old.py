import torch

from net_old import NetOld


def train_old(params, *, gradient_type, implementation='old'):

    torch.manual_seed(params['seed'])

    # teacher network
    model_target = NetOld()

    # student network
    model = NetOld()

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

            # compute normal gradient
            y = model(x)
            loss = torch.mean((y - y_target) ** 2)
            model.zero_grad()
            loss.backward()
            # watch the sign!
            G_l0_i = -model.weight_l0.grad.data.clone().t()
            G_l0_0 = -model.bias_l0.grad.data.clone()
            G_l1_i = -model.weight_l1.grad.data.clone().t()
            G_l1_0 = -model.bias_l1.grad.data.clone()

            # compute quasi-diagonal fisher information matrix
            phi_l0 = torch.zeros((len(x), 2))
            phi_l1 = torch.zeros((len(x), 2))
            for o in range(2):  # backpropagating transfer rates for each output
                model.zero_grad()
                y_transfer_rates = model.forward_transfer_rates(x)
                y_transfer_rates.data[:, o] = 1.

                for b in range(len(y_transfer_rates[:, o])):
                    y_transfer_rates[b, o].backward(retain_graph=True)
                    phi_l0 += model.activity_l0_grad ** 2
                    phi_l1 += model.activity_l1_grad ** 2

            # drop r^2 since linear activation function
            F_l0_00 = phi_l0.mean(dim=0)
            F_l0_0j = torch.einsum('bj,bk->bjk', model.activity_input, phi_l0).mean(dim=0)
            F_l0_ij = torch.einsum('bi,bj,bk->bijk', model.activity_input, model.activity_input, phi_l0).mean(dim=0)
            F_l1_00 = phi_l1.mean(dim=0)
            F_l1_0j = torch.einsum('bj,bk->bjk', model.activity_l0, phi_l1).mean(dim=0)
            F_l1_ij = torch.einsum('bi,bj,bk->bijk', model.activity_l0, model.activity_l0, phi_l1).mean(dim=0)

            # watch the sign!
            for k in range(2):

                sub_sum = 0.
                for i in range(3):
                    model.weight_l0.grad.data[k, i] = -1. * ((G_l0_i[i, k] * F_l0_00[k] - G_l0_0[k] * F_l0_0j[i, k]) / (F_l0_ij[i, i, k] * F_l0_00[k] - F_l0_0j[i, k] ** 2))
                    sub_sum += F_l0_0j[i, k] / F_l0_00[k] * model.weight_l0.grad.data[k, i]

                model.bias_l0.grad.data[k] = -1. * (G_l0_0[k] / F_l0_00[k] - sub_sum)

            for k in range(2):

                sub_sum = 0.
                for i in range(2):
                    model.weight_l1.grad.data[k, i] = -1. * ((G_l1_i[i, k] * F_l1_00[k] - G_l1_0[k] * F_l1_0j[i, k]) / (F_l1_ij[i, i, k] * F_l1_00[k] - F_l1_0j[i, k] ** 2))
                    sub_sum += F_l1_0j[i, k] / F_l1_00[k] * model.weight_l1.grad.data[k, i]

                model.bias_l1.grad.data[k] = -1. * (G_l1_0[k] / F_l1_00[k] - sub_sum)

        else:
            raise NotImplementedError()

        optimizer.step()

        history_loss[step] = loss.item()

    return history_loss

