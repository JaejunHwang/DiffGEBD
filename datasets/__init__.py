import copy
import os

from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data.dataloader import default_collate

from utils.sampler import DistBalancedBatchSampler
from .dataset import GEBDDataset, TAPOSDataset

# train
# True: train data / train
# False: val data / val
def build_dataloader(cfg, args, dataset_splits, train):
    assert len(dataset_splits) >= 1

    def build_dataset(dataset):
        name, split = dataset.split('_')

        ROOT = {
            'GEBD': os.getenv('GEBD_ROOT', args.gebd_data_dir),
            'TAPOS': os.getenv('TAPOS_ROOT', args.tapos_data_dir),
        }

        datasets = {
            'GEBD': {
                'train': 'train',
                'val': 'val',
            },
            'TAPOS': {
                'train': 'rgb',
                'val': 'rgb'
            }
        }

        assert name in datasets, f'Dataset {name} not exists!'
        root = os.path.join(ROOT[name], datasets[name][split])
        template = 'img_{:05d}.jpg' if name == 'GEBD' else 'img_{:05d}.jpg'

        if name == 'GEBD':
            dataset = GEBDDataset(cfg, root=root,
                                name=name,
                                split=split,
                                template=template,
                                train=train)
            
            
        elif name == 'TAPOS':
            dataset = TAPOSDataset(cfg, root=root,
                                name=name,
                                split=split,
                                template=template,
                                train=train)
            
        return dataset

    datasets = []
    for split in dataset_splits:
        datasets.append(build_dataset(split))

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        annotations = []
        for dataset in datasets:
            annotations.extend(copy.deepcopy(dataset.annotations))
        dataset = ConcatDataset(datasets)
        dataset.annotations = annotations

    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
        

    # collate_fn = (lambda x: x) if cfg.INPUT.END_TO_END else default_collate
    collate_fn = default_collate
    loader = DataLoader(dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                        sampler=sampler,
                        drop_last=False,
                        collate_fn=collate_fn,
                        num_workers=cfg.SOLVER.NUM_WORKERS)
    return loader
