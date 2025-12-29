from __future__ import absolute_import
import numpy as np
import torch
import random
from collections import defaultdict
from torch.utils.data import Sampler

#class RandomIdentitySampler(object):
class RandomIdentitySampler(torch.utils.data.sampler.Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        # for index, (_, gid, _, pid, _) in enumerate(data_source):
        for index, (_, gid, _, pid, _, _) in enumerate(data_source):

        #for index, (_, gid, _, _, _,_) in enumerate(data_source):
            self.index_dic[gid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances

# GroupPKBatchSampler
class GroupPKBatchSampler(Sampler):
    """
    Group PK Batch Sampler for Group Re-Identification.

    Each batch contains:
        P different group IDs
        K instances per group

    Total batch size = P * K
    """

    def __init__(self, dataset, P=8, K=4, drop_last=True):
        self.dataset = dataset
        self.P = P
        self.K = K
        self.drop_last = drop_last

        # Build index dictionary: group_id -> list of sample indices
        self.group_dict = defaultdict(list)
        for idx in range(len(dataset)):
            _, gid, _, _, _, _ = dataset[idx]
            self.group_dict[gid].append(idx)

        self.group_ids = list(self.group_dict.keys())

        # Number of batches per epoch
        self.num_batches = len(self.group_ids) // self.P
        if not drop_last and len(self.group_ids) % self.P != 0:
            self.num_batches += 1

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # Shuffle group IDs every epoch
        random.shuffle(self.group_ids)

        batch = []

        for gid in self.group_ids:
            indices = self.group_dict[gid]

            # Sample K instances from this group
            if len(indices) >= self.K:
                sampled = random.sample(indices, self.K)
            else:
                sampled = random.choices(indices, k=self.K)

            batch.extend(sampled)

            # When batch is full, yield it
            if len(batch) == self.P * self.K:
                yield batch
                batch = []

        # Handle last incomplete batch
        if len(batch) > 0 and not self.drop_last:
            yield batch
