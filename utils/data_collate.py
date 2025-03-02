from functools import partial

import numpy as np
import torch

from model.ops import grid_subsample, radius_search
# from utils.torch import build_dataloader

def precompute_data(points, lengths, num_stages, voxel_size, radius, neighbor_limits):
        """
        Precompute data for registration in stack mode.
        """
        assert num_stages == len(neighbor_limits)

        points_list = []
        lengths_list = []
        neighbors_list = []
        subsampling_list = []
        upsampling_list = []

        # grid subsampling
        for i in range(num_stages):
            if i > 0:
                points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
            points_list.append(points)
            lengths_list.append(lengths)
            voxel_size *= 2

        # radius search
        for i in range(num_stages):
            cur_points = points_list[i]
            cur_lengths = lengths_list[i]

            neighbors = radius_search(
                cur_points,
                cur_points,
                cur_lengths,
                cur_lengths,
                radius,
                neighbor_limits[i],
            )
            neighbors_list.append(neighbors)

            if i < num_stages - 1:
                sub_points = points_list[i + 1]
                sub_lengths = lengths_list[i + 1]

                subsampling = radius_search(
                    sub_points,
                    cur_points,
                    sub_lengths,
                    cur_lengths,
                    radius,
                    neighbor_limits[i],
                )
                subsampling_list.append(subsampling)

                upsampling = radius_search(
                    cur_points,
                    sub_points,
                    cur_lengths,
                    sub_lengths,
                    radius * 2,
                    neighbor_limits[i + 1],
                )
                upsampling_list.append(upsampling)

            radius *= 2

        return points_list, lengths_list, neighbors_list, subsampling_list, upsampling_list



def batch_collate_fn(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    batch_size = len(data_dicts)

    collate_key_list = ['pcd_points', 'lengths', 'neighbors', 'subsampling', 'upsampling']
    # raw_key_list = []
   
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items(): 
            if key not in collated_dict:
                collated_dict[key] = []
            if key in collate_key_list:
                collated_dict[key].append(value)
            else:
                tensor1 = value.unsqueeze(0)
                collated_dict[key].append(tensor1)
            
    for key, value in collated_dict.items():
        if key not in collate_key_list:
            collated_dict[key] = torch.cat(value, dim=0)
    collated_dict['batch_size'] = torch.IntTensor(batch_size)
    return collated_dict

def calibrate_neighbors(
    dataset, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=2000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))   
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)

    for i in range(len(dataset)):
        data_dict = dataset[i]
        
        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits

def build_dataloader_stack_mode(
    dataset,
    collate_fn,
    num_stages,
    voxel_size,
    search_radius,
    neighbor_limits,

    batch_size=1,
    num_workers=1,
    shuffle=False,
    drop_last=False,
    pin_memory=False,
    precompute_data=True,
):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            search_radius=search_radius,
            neighbor_limits=neighbor_limits,
            precompute_data=precompute_data,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=None,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
    return dataloader
