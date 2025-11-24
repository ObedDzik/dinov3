# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extended for custom medical imaging datasets

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger("dinov3")


# class MicrousDataset(Dataset):
#     def __init__(
#         self,
#         root: str,
#         split: str = "train",
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         transforms: Optional[Callable] = None,
#         **kwargs,
#     ):
#         self.root = Path(root)
#         self.split = split
#         self.transform = transform
#         self.target_transform = target_transform
#         self.transforms = transforms
        
#         self.image_paths = self._load_image_paths()
#         logger.info(f"Loaded {len(self.image_paths)} images for SSL training")
    
#     def _load_image_paths(self):
#         image_paths = list(Path(self.root).rglob("*.png"))
#         return image_paths
    
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = Image.open(image_path).convert('RGB')
        
#         # Dummy target (will be ignored by target_transform=lambda _: ())
#         target = 0
        
#         if self.transforms is not None:
#             image, target = self.transforms((image, target))
#         else:
#             if self.transform is not None:
#                 image = self.transform(image)
#             if self.target_transform is not None:
#                 target = self.target_transform(target)
        
#         return image, target
    
#     def _load_sample(self, sample):
#         """
#         Load a single sample. Override if needed.
        
#         Args:
#             sample: A sample from self.samples
            
#         Returns:
#             (image, target) tuple
#         """
#         if isinstance(sample, (str, Path)):
#             image = Image.open(sample).convert('RGB')
#             target = 0
#         elif isinstance(sample, tuple):
#             image_path, target = sample
#             image = Image.open(image_path).convert('RGB')
#         elif isinstance(sample, dict):
#             image = sample.get('image')
#             target = sample.get('label', 0)
#             if isinstance(image, (str, Path)):
#                 image = Image.open(image).convert('RGB')
#         else:
#             raise ValueError(f"Unsupported sample type: {type(sample)}")
        
#         return image, target


# class DictDatasetAdapter(Dataset):
#     """
#     Adapter for datasets that return dictionaries (like {'image': ..., 'label': ...})
#     to work with DINOv3's expected (image, target) tuple format.
#     """
    
#     def __init__(
#         self,
#         base_dataset: Dataset,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         transforms: Optional[Callable] = None,
#     ):
#         """
#         Args:
#             base_dataset: Your existing dataset that returns dicts
#             transform: Transform to apply to images
#             target_transform: Transform to apply to targets
#             transforms: Combined transform (takes precedence)
#         """
#         self.base_dataset = base_dataset
        
#         if transforms is not None:
#             self.transforms = transforms
#             self.transform = None
#             self.target_transform = None
#         else:
#             self.transform = transform
#             self.target_transform = target_transform
#             self.transforms = None
    
#     def __len__(self) -> int:
#         return len(self.base_dataset)
    
#     def __getitem__(self, idx: int) -> Tuple[Any, Any]:
#         sample = self.base_dataset[idx]
        
#         # Extract image and target from dict
#         if isinstance(sample, dict):
#             image = sample.get('image')
#             target = sample.get('label', 0)
#         else:
#             # Already a tuple
#             image, target = sample
        
#         # Apply transforms
#         if self.transforms is not None:
#             image, target = self.transforms((image, target))
#         else:
#             if self.transform is not None:
#                 image = self.transform(image)
#             if self.target_transform is not None:
#                 target = self.target_transform(target)
        
#         return image, target
    
class MicrousDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        **kwargs,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        
        self.image_paths = self._load_image_paths()
        logger.info(f"Loaded {len(self.image_paths)} images for SSL training")
    
    def _load_image_paths(self):
        image_paths = list(Path(self.root).rglob("*.png"))
        return sorted(image_paths)  # Sort for reproducibility
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        img = None
        try:
            img = Image.open(image_path)
            img.load()
            image = img.convert('RGB')
            image = image.copy()
        finally:
            if img is not None:
                img.close()
        target = 0
        if self.transforms is not None:
            image, target = self.transforms((image, target))
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
        
        return image, target