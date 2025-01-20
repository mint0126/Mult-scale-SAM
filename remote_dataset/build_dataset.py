from .loveda import LoveDA
from torch.utils.data import DataLoader
from .potsdam import Potsdam
from .vaihingen import Vaihingen


def prepare_dataset(dataset, train_images_path, train_annotations, val_images_path, val_annotations, 
                    batch_size=32, transform=None,target_transform=None):

    
    if dataset == 'loveda':
        train_set = LoveDA(train_images_path, train_annotations, transform, target_transform)
        val_set = LoveDA(val_images_path, val_annotations, transform, target_transform)
    elif dataset == 'nwpu':
        train_set = CocoDataset(train_images_path, train_annotations, 
                            transform = transform, target_transform = target_transform)
        val_set = CocoDataset(val_images_path, val_annotations, 
                            transform = transform, target_transform = target_transform)
    elif dataset == 'potsdam':
        train_set = Potsdam(train_images_path, train_annotations, 
                            transforms = transform, target_transforms = target_transform)
        val_set = Potsdam(val_images_path, val_annotations, 
                            transforms = transform, target_transforms = target_transform)
    elif dataset == 'vaihingen':
        train_set = Vaihingen(train_images_path, train_annotations, 
                            transforms = transform, target_transforms = target_transform)
        val_set = Vaihingen(val_images_path, val_annotations, 
                            transforms = transform, target_transforms = target_transform)
    
    image_sets = {'train': train_set,'val':val_set}
    dataloaders = {
        'train' : DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
        }


    return dataloaders, image_sets
