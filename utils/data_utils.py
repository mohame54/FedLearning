import torch
import os
import copy
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


class PemDataset(Dataset):
  def __init__(
      self,
      recent_data,
      day_data,
      week_data,
      target_data,
      norms_data=None,
  ):
      self.recent_data = recent_data.astype(np.float32)
      self.day_data = day_data.astype(np.float32)
      self.week_data = week_data.astype(np.float32)
      self.targets = target_data.astype(np.float32)
      self.normalize_data = norms_data is not None
      self.norms_data = norms_data

  def __len__(self):
      return len(self.recent_data)

  def normalize(self, data):
      mean, std = [torch.tensor(dt) for dt in self.norms_data]
      eps = torch.full_like(std, torch.finfo(data.dtype).eps)
      return (data - mean) / torch.maximum(std, eps)

  def __getitem__(self, idx):
      recent = torch.from_numpy(self.recent_data[idx]).to(torch.float32).transpose(-1, 0)
      day = torch.from_numpy(self.day_data[idx]).to(torch.float32).transpose(-1, 0)
      week = torch.from_numpy(self.week_data[idx]).to(torch.float32).transpose(-1, 0)
      target = torch.from_numpy(self.targets[idx]).to(torch.float32).transpose(-1, 0)
      if self.normalize_data:
          recent = self.normalize(recent)
          day = self.normalize(day)
          week = self.normalize(week)
      return recent, day, week, target

  @classmethod
  def from_data_directory(
      cls,
      data_path,
      keyword="train",
      normalize=False,
  ):
      data_path = os.path.join(data_path, f"{keyword}.npz")
      data = np.load(data_path)
      recent_data = data['hour']
      day_data = data['day']
      week_data = data['week']
      targets = data['target']
      norms_data = None
      if normalize:
          norms_file = os.path.join(data_path, "norms.npz")
          norms_data = np.load(norms_file)
          norms_data = [norms_data['mean'], norms_data['std']]

      return cls(
          recent_data,
          day_data,
          week_data,
          targets,
          norms_data=norms_data,
     )


def split_data_partitions(dataset, patitions, seed=0):
    return random_split(dataset, patitions, torch.Generator().manual_seed(seed))


def prepare_data(
    dataset_folder_pth,
    num_partitions,
    train_batch_size=128,
    val_batch_size=64,
    seed=0,
    **dataloader_kwargs,
):
    train_ds = PemDataset.from_data_directory(dataset_folder_pth)
    val_ds = PemDataset.from_data_directory(dataset_folder_pth, keyword='val')
    train_sz_parts = [len(train_ds) // num_partitions] * num_partitions
    train_sz_parts[-1] = train_sz_parts[-1] + len(train_ds) % num_partitions
    val_sz_parts = [len(val_ds) // num_partitions] * num_partitions
    val_sz_parts[-1] = val_sz_parts[-1] + len(val_ds) % num_partitions
    
    train_parts = split_data_partitions(train_ds, train_sz_parts, seed)
    val_parts = split_data_partitions(val_ds, val_sz_parts, seed)
    train_loaders, val_loaders = [], []
    val_dataloader_kwargs = copy.deepcopy(dataloader_kwargs)
    val_dataloader_kwargs['shuffle'] = False
    for train_ds_part, val_ds_part in zip(train_parts, val_parts):
        train_loaders.append(
            DataLoader(
                train_ds_part,
                batch_size=train_batch_size,
                **dataloader_kwargs
            )
        )
        val_loaders.append(
            DataLoader(
                val_ds_part,
                batch_size=val_batch_size,
                **val_dataloader_kwargs
            )
        )
    test_ds = PemDataset.from_data_directory(dataset_folder_pth, keyword='val')
    test_loader =  DataLoader(
                    test_ds,
                    batch_size=val_batch_size,
                    **val_dataloader_kwargs
                )
    return train_loaders, val_loaders, test_loader