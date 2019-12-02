import math
import torch


class NaturalGradientLayer(torch.nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

        self.G_i = None
        self.G_0 = None
        self.phi = None
        self.activity_input = None
        self.activity_output = None
        self.activity_output_grad = None

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def _backward_hook_activity(self, grad):
        self.activity_output_grad = grad.clone()

    def forward_transfer_rates(self, input):
        self.activity_input = input
        self.activity_output = torch.nn.functional.linear(self.activity_input, self.weight.data, self.bias.data)
        self.activity_output.register_hook(self._backward_hook_activity)
        return self.activity_output

    def compute_G(self):
        # watch the sign!
        self.G_i = -self.weight.grad.data.clone().t()
        self.G_0 = -self.bias.grad.data.clone()

    def reset_phi(self):
        self.phi = None

    def update_phi(self):
        if self.phi is None:
            self.phi = torch.zeros_like(self.activity_output)
        self.phi += self.activity_output_grad ** 2

    def compute_F(self):
        # drop r^2 since linear activation function
        self.F_00 = self.phi.mean(dim=0)
        self.F_0j = torch.einsum('bj,bk->bjk', self.activity_input, self.phi).mean(dim=0)
        self.F_ij = torch.einsum('bi,bj,bk->bijk', self.activity_input, self.activity_input, self.phi).mean(dim=0)

    def compute_grad(self):
        # watch the sign!
        for k in range(self.out_features):

            sub_sum = 0.
            for i in range(self.in_features):
                self.weight.grad.data[k, i] = -1. * ((self.G_i[i, k] * self.F_00[k] - self.G_0[k] * self.F_0j[i, k]) / (self.F_ij[i, i, k] * self.F_00[k] - self.F_0j[i, k] ** 2))
                sub_sum += self.F_0j[i, k] / self.F_00[k] * self.weight.grad.data[k, i]

            self.bias.grad.data[k] = -1. * (self.G_0[k] / self.F_00[k] - sub_sum)


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.in_features = 3
        self.out_features = 2

        self.layer0 = NaturalGradientLayer(self.in_features, 2)
        self.layer1 = NaturalGradientLayer(2, self.out_features)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        return x

    def forward_transfer_rates(self, x):
        x.requires_grad = True
        x = self.layer0.forward_transfer_rates(x)
        x = self.layer1.forward_transfer_rates(x)
        return x

    def compute_natural_grad(self, x, y_target):

        # compute normal gradient
        y = self(x)
        loss = torch.mean((y - y_target) ** 2)
        self.zero_grad()
        loss.backward()

        # compute quasi-diagonal fisher information matrix (Ollivier 2015)
        self.layer0.compute_G()
        self.layer1.compute_G()

        self.layer0.reset_phi()
        self.layer1.reset_phi()
        for o in range(self.out_features):  # compute backpropagation transfer rates for each output

            self.zero_grad()
            y_transfer_rates = self.forward_transfer_rates(x)
            y_transfer_rates.data[:, o] = 1.

            # iterate over all batch samples
            for b in range(len(y_transfer_rates[:, o])):
                y_transfer_rates[b, o].backward(retain_graph=True)
                self.layer0.update_phi()
                self.layer1.update_phi()

        self.layer0.compute_F()
        self.layer1.compute_F()

        self.layer0.compute_grad()
        self.layer1.compute_grad()

        return loss
