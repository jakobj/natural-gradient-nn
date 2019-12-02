import math
import torch


class NetOld(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.weight_l0 = torch.nn.Parameter(torch.zeros(2, 3))
        self.bias_l0 = torch.nn.Parameter(torch.zeros(2))
        self.weight_l1 = torch.nn.Parameter(torch.zeros(2, 2))
        self.bias_l1 = torch.nn.Parameter(torch.zeros(2))

        # self.activity_input = torch.empty(3)
        # self.activity_l0 = torch.empty(2)
        # self.activity_l1 = torch.empty(2)
        # self.activity_l0_grad = torch.empty(2)
        # self.activity_l1_grad = torch.empty(2)

        self.reset_parameters(self.weight_l0, self.bias_l0)
        self.reset_parameters(self.weight_l1, self.bias_l1)

    def reset_parameters(self, weight, bias):
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(bias, -bound, bound)

    def _backward_hook_activity_l0(self, grad):
        self.activity_l0_grad = grad

    def _backward_hook_activity_l1(self, grad):
        self.activity_l1_grad = grad

    def forward(self, x):
        x = torch.mm(x, self.weight_l0.t()) + self.bias_l0
        x = torch.mm(x, self.weight_l1.t()) + self.bias_l1
        return x

    def forward_transfer_rates(self, x):

        # store input and mark as requires grad to allow backprop on activities
        self.activity_input = x
        self.activity_input.requires_grad = True

        # store activities in forward pass
        self.activity_l0 = torch.mm(self.activity_input, self.weight_l0.data.t()) + self.bias_l0.data
        # track gradient on pytorch's backward path
        self.activity_l0.register_hook(self._backward_hook_activity_l0)

        # store activities in forward pass
        self.activity_l1 = torch.mm(self.activity_l0, self.weight_l1.data.t()) + self.bias_l1.data
        # track gradient on pytorch's backward path
        self.activity_l1.register_hook(self._backward_hook_activity_l1)

        return self.activity_l1
