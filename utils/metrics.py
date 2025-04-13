import numpy as np
import torch


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def detransform(data, norms):
    mean = norms["mean"]
    std = norms["std"]
    data = data * std + mean
    return data


def mae_loss(output, targets, norms_params=None):
    output = output.transpose(1, 3)
    tars = targets[:, 0, :,:].unsqueeze(dim=1)
    if norms_params is not None:
        norms_params = {k:v.to(output.device) for k, v in norms_params.items()}
        tars = detransform(tars, norms_params)
        output = detransform(output, norms_params)
    return masked_mae(output, tars, 0.0)


def mape_error(output, targets, norms_params=None):
    output = output.transpose(1, 3)
    tars = targets[:, 0, :,:].unsqueeze(dim=1)
    if norms_params is not None:
        norms_params = {k:v.to(output.device) for k, v in norms_params.items()}
        tars = detransform(tars, norms_params)
        output = detransform(output, norms_params)
    return masked_mape(output, tars, 0.0)


def rmse_error(output, targets, norms_params=None):
    output = output.transpose(1, 3)
    tars = targets[:, 0, :,:].unsqueeze(dim=1)
    if norms_params is not None:
        norms_params = {k:v.to(output.device) for k, v in norms_params.items()}
        tars = detransform(tars, norms_params)
        output = detransform(output, norms_params)
    return torch.sqrt(((output - tars) ** 2).mean())