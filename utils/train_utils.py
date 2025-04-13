from typing import List
import numpy as np
from collections import OrderedDict
import torch
import inspect
from tqdm import tqdm
from .metrics import mae_loss, rmse_error, mape_error 


Mean = lambda x: sum(x) / len(x)


def create_opt(model, weight_decay=0.1,lr=1e-4,betas=[0.9, 0.95], eps=1e-8):
    params_dict = {nm: p for nm, p in model.named_parameters() if p.requires_grad}
    to_decay = [ p for nm, p in params_dict.items() if p.dim() >=2 ]
    no_decay = [ p for nm, p in params_dict.items() if p.dim() <2 ]
    groups = [
        {"params": to_decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]
    fused = "fused" in inspect.signature(torch.optim.Adam).parameters
    optim = torch.optim.AdamW(groups, lr, betas, eps=eps, fused=fused)
    print(f"to decay: {sum([p.numel() for p in to_decay])} parameters no decay: {sum([p.numel() for p in no_decay])} parmerers.")
    return optim


def get_parameters(net) -> List[np.ndarray]:
    return net.state_dict()


def set_parameters(net, parameters):
  net.load_state_dict(parameters, strict=True)



def train_epoch(
    model,
    train_ds,
    opt,
    device,
    scaler=None,
    max_norm=None,
    grad_accumelation=1,
    norms_params=None,
 
):
      model.train()
      losses = []
      all_preds, all_tars = [], []
      rmses, mapes = [], []
      loop = tqdm(train_ds, desc="Training loop")
      loss_accum = 0.0
      opt.zero_grad()
      for i, (rec, day, week, targets) in enumerate(loop):
          rec, day, week = rec.to(device), day.to(device), week.to(device)
          targets = targets.to(device)
          if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(rec, day, week)
                loss = mae_loss(pred, targets)
            loss = loss / grad_accumelation
            scaler.scale(loss).backward()
          else:
              pred = model(rec, day, week)
              loss = mae_loss(pred, targets)
              loss = loss / grad_accumelation
              loss.backward()
          all_preds.append(pred.cpu())
          all_tars.append(targets.cpu())
          # update the accumelation logs
          loss_accum = loss_accum + loss.item()
          if (i+1) % grad_accumelation == 0:
            if max_norm is not None:
              scaler.unscale_(opt)
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            opt.zero_grad()
            losses.append(loss_accum)
            loss_accum = 0.0

           
            to_show = {
                  f"Training loss":Mean(losses),
              }
            if norms_params is not None:
                mape = mape_error(pred, targets, norms_params).item()
                rmse = rmse_error(pred, targets, norms_params).item()
                rmses.append(rmse)
                mapes.append(mape)
                to_show["rmse"] = Mean(rmses)
                to_show["mape"] = Mean(mapes)
            loop.set_postfix(to_show)
            del rec, day, week , targets, pred
            torch.cuda.empty_cache()
      all_preds = torch.cat(all_preds)
      all_tars = torch.cat(all_tars)
      loss = mae_loss(all_preds, all_tars, norms_params).item()
      mape = mape_error(all_preds, all_tars, norms_params).item()
      rmse = rmse_error(all_preds, all_tars, norms_params).item()
      return {'loss':loss, "mape":mape, "rmse":rmse}


def val_epoch(model,val_ds,device, norms_params=None):
    model.eval()
    losses, rmses, mapes = [],[],[]
    all_preds, all_tars = [], []
    with torch.no_grad():
        loop = tqdm(val_ds, desc="Validation loop")
        for rec, day, week, targets in loop:
          rec, day, week = rec.to(device), day.to(device), week.to(device)
          targets = targets.to(device)
          with torch.autocast(device_type='cuda', dtype=torch.float16):
            pred = model(rec, day, week)
            loss = mae_loss(pred, targets, norms_params)
          all_preds.append(pred.cpu())
          all_tars.append(targets.cpu())
          loss = loss.item()
          losses.append(loss)
          to_show = {
              "loss":Mean(losses)
          }
          if norms_params is not None:
            mape = mape_error(pred, targets, norms_params).item()
            rmse = rmse_error(pred, targets, norms_params).item()
            rmses.append(rmse)
            mapes.append(mape)
            to_show["rmse"] = Mean(rmses)
            to_show["mape"] = Mean(mapes)
          loop.set_postfix(to_show)
          del rec, day, week , targets, pred
          torch.cuda.empty_cache()
    if norms_params is None:
      return Mean(losses)
    all_preds = torch.cat(all_preds)
    all_tars = torch.cat(all_tars)
    loss = mae_loss(all_preds, all_tars, norms_params).item()
    if norms_params is None:
       return loss
    mape = mape_error(all_preds, all_tars, norms_params).item()
    rmse = rmse_error(all_preds, all_tars, norms_params).item()
    return {'loss':loss, "mape":mape, "rmse":rmse}
