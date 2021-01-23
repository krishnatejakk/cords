import os
from torchmeta.modules import MetaModule
import torch.nn.functional as F
from tqdm import tqdm
import logging
from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters
from cords.utils.models import MiniImagenetNetwork, FrozenMiniImagenetNetwork
from collections import OrderedDict
import random
from itertools import combinations
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torchmeta.utils.data.dataset import CombinationMetaDataset
import torch
from collections import OrderedDict
from torchmeta.modules import MetaModule
import copy


def flatten_params(param_list):
    l = [torch.flatten(p) for p in param_list]
    flat = torch.cat(l)
    return flat


# def gradient_update_parameters(model,
#                                loss,
#                                params=None,
#                                step_size=0.5,
#                                first_order=False):
#     """Update of the meta-parameters with one step of gradient descent on the
#     loss function.
#
#     Parameters
#     ----------
#     model : `torchmeta.modules.MetaModule` instance
#         The model.
#
#     loss : `torch.Tensor` instance
#         The value of the inner-loss. This is the result of the training dataset
#         through the loss function.
#
#     params : `collections.OrderedDict` instance, optional
#         Dictionary containing the meta-parameters of the model. If `None`, then
#         the values stored in `model.meta_named_parameters()` are used. This is
#         useful for running multiple steps of gradient descent as the inner-loop.
#
#     step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
#         The step size in the gradient update. If an `OrderedDict`, then the
#         keys must match the keys in `params`.
#
#     first_order : bool (default: `False`)
#         If `True`, then the first order approximation of MAML is used.
#
#     Returns
#     -------
#     updated_params : `collections.OrderedDict` instance
#         Dictionary containing the updated meta-parameters of the model, with one
#         gradient update wrt. the inner-loss.
#     """
#     if not isinstance(model, MetaModule):
#         raise ValueError('The model must be an instance of `torchmeta.modules.'
#                          'MetaModule`, got `{0}`'.format(type(model)))
#
#     if params is None:
#         params = OrderedDict(model.meta_named_parameters())
#
#     grads = torch.autograd.grad(loss,
#                                 params.values(),
#                                 create_graph=not first_order,
#                                 allow_unused=True)
#
#     updated_params = OrderedDict()
#
#     if isinstance(step_size, (dict, OrderedDict)):
#         for (name, param), grad in zip(params.items(), grads):
#             updated_params[name] = param - step_size[name] * grad
#     else:
#         for (name, param), grad in zip(params.items(), grads):
#             if grad is None:
#                 updated_params[name] = param
#             else:
#                 updated_params[name] = param - step_size * grad
#
#     return updated_params


class CustomSequentialSampler(SequentialSampler):
    def __init__(self, data_source, batches):
        if not isinstance(data_source, CombinationMetaDataset):
            raise TypeError('Expected `data_source` to be an instance of '
                            '`CombinationMetaDataset`, but found '
                            '{0}'.format(type(data_source)))
        super(CustomSequentialSampler, self).__init__(data_source)
        self.cnt = 0
        self.batches =batches

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        num_classes_per_task = self.data_source.num_classes_per_task
        for _ in combinations(range(num_classes), num_classes_per_task):
            self.cnt += 1
            self.cnt = self.cnt % len(self.batches)
            yield self.batches[self.cnt]

    def __update_batches__(self, batches):
        self.cnt = 0
        self.batches = batches


def sampler(num_classes, num_classes_per_task, num_batches):
  batches = []
  for _ in range(num_batches):
      batches.append(tuple(random.sample(range(num_classes), num_classes_per_task)))
  return batches


def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())


logger = logging.getLogger(__name__)

def compute_meta_gradients(model, dataloader):
    with tqdm(dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()
            #data = dataset[batch]
            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)
            #outer_loss = torch.tensor(0., device=args.device)
            #accuracy = torch.tensor(0., device=args.device)
            cnt = 0
            for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):
                #for i in range(1):
                train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)
                model.zero_grad()
                params = gradient_update_parameters(model,
                                                inner_loss,
                                                step_size=args.step_size,
                                                first_order=args.first_order)

                test_logit = model(test_input, params=params)
                outer_loss = F.cross_entropy(test_logit, test_target)
                outer_loss.backward()
                if cnt == 0:
                    with torch.no_grad():
                        curr_params = list(model.parameters())[-2:]
                        curr_grads = [param.grad.data for param in curr_params]
                        curr_grads = flatten_params(curr_grads)
                    per_elem_grads = curr_grads.view(1, -1)
                    cnt = 1
                else:
                    with torch.no_grad():
                        curr_params = list(model.parameters())[-2:]
                        curr_grads = [param.grad.data for param in curr_params]
                        curr_grads = flatten_params(curr_grads)
                    per_elem_grads = torch.cat([per_elem_grads, curr_grads.view(1, -1)], dim=0)
        print()


def train(args):
    logger.warning('This script is an example to showcase the MetaModule and '
                   'data-loading features of Torchmeta, and as such has been '
                   'very lightly tested. For a better tested implementation of '
                   'Model-Agnostic Meta-Learning (MAML) using Torchmeta with '
                   'more features (including multi-step adaptation and '
                   'different datasets), please check `https://github.com/'
                   'tristandeleu/pytorch-maml`.')

    dataset = miniimagenet(args.folder,
                       shots=args.num_shots,
                       ways=args.num_ways,
                       shuffle=True,
                       test_shots=15,
                       meta_train=True,
                       download=args.download)

    model = MiniImagenetNetwork(3, args.num_ways, hidden_size=args.hidden_size)
    model1 = FrozenMiniImagenetNetwork(3, args.num_ways, hidden_size=args.hidden_size)
    cached_state_dict = copy.deepcopy(model.state_dict())
    model1.load_state_dict(cached_state_dict)
    model.to(device=args.device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batches = sampler(num_classes=len(dataset.dataset), num_classes_per_task=args.num_ways, num_batches=20000)
    custom_sampler = CustomSequentialSampler(dataset, batches)
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     #sampler=custom_sampler,
                                     num_workers=args.num_workers)

    #compute_meta_gradients(model1, dataloader)
    # Training loop
    with tqdm(dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()
            #data = dataset[batch]
            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
            for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):
                train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)

                model.zero_grad()
                params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    step_size=args.step_size,
                                                    first_order=args.first_order)

                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

            outer_loss.div_(args.batch_size)
            accuracy.div_(args.batch_size)

            outer_loss.backward()
            meta_optimizer.step()

            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
            if batch_idx >= args.num_batches:
                break

    # Save model
    if args.output_folder is not None:
        filename = os.path.join(args.output_folder, 'maml_omniglot_'
            '{0}shot_{1}way.th'.format(args.num_shots, args.num_ways))
        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')

    parser.add_argument('--folder', type=str,  default='../../../../data',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')

    parser.add_argument('--first-order', action='store_true',
        help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.4,
        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=5,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batches the model is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true', default=True,
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true',
        help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')

    train(args)