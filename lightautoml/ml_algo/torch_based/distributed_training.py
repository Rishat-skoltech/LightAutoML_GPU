import torch.multiprocessing as mp
import torch

from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Union

from copy import deepcopy
import cupy as cp
import torch.optim as optim

def train_model(rank, model, data, y, weights, c, cat_idx, self_loss, opt_params, queue, ready):
    if rank is None:
        ind = 0
        size = 1
        device_id = f'cuda:0'
    else:
        ind = rank
        size = torch.cuda.device_count()
        device_id = f'cuda:{rank}'

    if type(data) == cp.ndarray:
        data = cp.copy(data)
    else:
        data = data.compute()

    size_base = data.shape[0] // size
    residue = int(data.shape[0] % size)
    offset = size_base * ind + min(ind, residue)

    if type(data) == cp.ndarray:
        data = data[offset:offset + size_base + int(residue > ind), :]
    else:
        data = data.iloc[offset:offset + size_base + int(residue > ind)]

    if y is not None:
        if type(y) != cp.ndarray:
            y = cp.copy(y.compute().values[offset:offset + size_base + int(residue > ind)])
        else:
            y = cp.copy(y[offset:offset + size_base + int(residue > ind)])
        y = torch.as_tensor(y.astype(cp.float32), device=device_id)
    if weights is not None:

        if type(weights) != cp.ndarray:
            weigths = cp.copy(weights.compute().values[offset:offset + size_base + int(residue > ind)])
        else:
            weights = cp.copy(weights[offset:offset + size_base + int(residue > ind)])

        weights = torch.as_tensor(weights.astype(cp.float32), device=device_id)

    if 0 < len(cat_idx) < data.shape[1]:
        # noinspection PyTypeChecker
        data_cat = torch.as_tensor(
            data[cat_idx].values.astype(cp.int32),
            device=device_id
        )
        data = torch.as_tensor(
            data[data.columns.difference(cat_idx)] \
            .values.astype(cp.float32),
            device=device_id
        )
    elif len(cat_idx) == 0:
        data = torch.as_tensor(data.values.astype(cp.float32), device=device_id)


    else:
        data_cat = torch.as_tensor(data.values.astype(cp.int32), device=device_id)

    model = deepcopy(model)
    model.to(rank)

    def _optimize(model, data: torch.Tensor,
                  data_cat: Optional[torch.Tensor], y: torch.Tensor = None,
                  weights: Optional[torch.Tensor] = None, c: float = 1, parameters=None):
        """Optimize single model.

        Args:
            data: Numeric data to train.
            data_cat: Categorical data to train.
            y: Target values.
            weights: Item weights.
            c: Regularization coefficient.

        """
        model.train()
        opt = optim.LBFGS(
            model.parameters(),
            lr=0.1,
            max_iter=parameters['max_iter'],
            tolerance_change=parameters['tol'],
            tolerance_grad=parameters['tol'],
            line_search_fn='strong_wolfe'
        )
        # keep history
        results = []
        def closure():
            opt.zero_grad()
            output = model(data, data_cat)
            loss = _loss_fn(model, y.reshape(-1, 1), output, weights, c, self_loss=self_loss)
            if loss.requires_grad:
                loss.backward()
            results.append(loss.item())
            return loss
        opt.step(closure)
        return model
    def _loss_fn(
        model,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        weights: Optional[torch.Tensor],
        c: float,
        self_loss=None
    ) -> torch.Tensor:
        """Weighted loss_fn wrapper.

        Args:
            y_true: True target values.
            y_pred: Predicted target values.
            weights: Item weights.
            c: Regularization coefficients.

        Returns:
            Loss+Regularization value.

        """
        # weighted loss
        loss = self_loss(y_true, y_pred, sample_weight=weights)

        n = y_true.shape[0]
        if weights is not None:
            n = weights.sum()

        all_params = torch.cat([y.view(-1) for (x, y) in model.named_parameters() if x != 'bias'])

        penalty = torch.norm(all_params, 2).pow(2) / 2 / n

        return loss + .5 * penalty / c

    model = _optimize(model, data, data_cat, y, weights, c, parameters=opt_params)
    queue.put(model.state_dict())
    ready.wait()
