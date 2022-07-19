from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def initialize_rnn_hidden_state(dim_sum,n_layers,n_params):
    h0 = Variable(torch.zeros(n_layers,n_params,dim_sum),requires_grad=False)
    h0.data = h0.data.cuda()
    return h0

class MetaOptimizerRNN(nn.Module):

    def __init__(self, model, num_layers, input_dim, hidden_size):
        super(MetaOptimizerRNN, self).__init__()
        self.meta_model = model
        self.first_order = nn.RNN(input_dim,hidden_size,num_layers,batch_first=True,bias=False)
        self.outputer = nn.Linear(hidden_size,1,bias=False)
        self.outputer.weight.data.mul_(0.1)
        self.hidden_size = hidden_size

    def parallel(self):
        self.first_order.cuda()
        self.outputer.cuda()

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)
        if keep_states > 0:
            self.h0 = Variable(self.h0.data)
        else:
            self.h0 = initialize_rnn_hidden_state(self.hidden_size, 1, self.meta_model.get_flat_params().size(0))

    def forward(self, x):
        output1, hn1 = self.first_order(x, self.h0)
        self.h0 = hn1
        o1 = self.outputer(output1)
        return o1.squeeze()

    def meta_update(self, model_with_grads, loss, eps=None):
        # First we need to create a flat version of parameters and gradients
        grads = []
        for name, module in model_with_grads.named_modules():
            # print(name)
            if len(module._parameters) != 0:
                grads.append(module._parameters['weight'].grad.data.view(-1).unsqueeze(-1))
                try:
                    grads.append(module._parameters['bias'].grad.data.view(-1).unsqueeze(-1))
                except:
                    pass


        flat_params = self.meta_model.get_flat_params()
        flat_grads = torch.cat(grads)
        second_moment = (flat_grads**2).mean() + 1e-5
        flat_grads = flat_grads / torch.sqrt(second_moment)
        inputs = Variable(flat_grads.view(-1, 1).unsqueeze(1))
        if eps is None:
            update = self(inputs)
            distance = None
        else:
            state_adv = inputs.detach() + 0.01 * torch.randn(inputs.shape).cuda().detach()
            step_size = eps / 5
            ori_h0 = self.h0

            for _ in range(10):
                state_adv.requires_grad_()
                with torch.enable_grad():
                    perturbed_update = self(state_adv)
                    self.h0 = ori_h0
                    # print(self.h0[0])
                    update = self(inputs)
                    self.h0 = ori_h0
                    # print(self.h0[0][1][1])
                    distance = F.mse_loss(perturbed_update, update)
                    grad = torch.autograd.grad(distance, [state_adv])[0]
                    state_adv = state_adv.detach() + step_size * torch.sign(grad.detach())
                    state_adv = torch.min(torch.max(state_adv, inputs - eps), inputs + eps)

            self.h0 = ori_h0
            perturbed_inputs = state_adv.detach()
            perturbed_update = self(perturbed_inputs)
            self.h0 = ori_h0
            update = self(inputs)
            distance = F.mse_loss(update, perturbed_update)
        flat_params = flat_params + update

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model, distance


# A helper class that keeps track of meta updates
# It's done by replacing parameters with variables and applying updates to
# them.
class MetaModel:

    def __init__(self, model):
        self.model = model


    def reset(self):
        for module in self.model.modules():
            if len(module._parameters) != 0:
                module._parameters['weight'] = Variable(module._parameters['weight'].data)
                try:
                    module._parameters['bias'] = Variable(module._parameters['bias'].data)
                except:
                    pass


    def get_flat_params(self):
        params = []
        for name, module in self.model.named_modules():
            if len(module._parameters) != 0:
                params.append(module._parameters['weight'].view(-1))
                try:
                    params.append(module._parameters['bias'].view(-1))
                except:
                    pass
        return torch.cat(params)


    def set_flat_params(self, flat_params):
        # Restore original shapes
        offset = 0
        for module in self.model.modules():
            if len(module._parameters) != 0:
                weight_shape = module._parameters['weight'].size()
                weight_flat_size = reduce(mul, weight_shape, 1)
                module._parameters['weight'] = flat_params[
                                               offset:offset + weight_flat_size].view(*weight_shape)
                try:
                    bias_shape = module._parameters['bias'].size()
                    bias_flat_size = reduce(mul, bias_shape, 1)
                    module._parameters['bias'] = flat_params[
                                                 offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(
                        *bias_shape)
                except:
                    bias_flat_size = 0
                offset += weight_flat_size + bias_flat_size


    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)
