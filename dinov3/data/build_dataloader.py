# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extended for custom medical imaging datasets

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger("dinov3")


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
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Dummy target (will be ignored by target_transform=lambda _: ())
        target = 0
        
        if self.transforms is not None:
            image, target = self.transforms((image, target))
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
        
        return image, target
    
    def _load_sample(self, sample):
        """
        Load a single sample. Override if needed.
        
        Args:
            sample: A sample from self.samples
            
        Returns:
            (image, target) tuple
        """
        if isinstance(sample, (str, Path)):
            image = Image.open(sample).convert('RGB')
            target = 0
        elif isinstance(sample, tuple):
            image_path, target = sample
            image = Image.open(image_path).convert('RGB')
        elif isinstance(sample, dict):
            image = sample.get('image')
            target = sample.get('label', 0)
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
        else:
            raise ValueError(f"Unsupported sample type: {type(sample)}")
        
        return image, target


class DictDatasetAdapter(Dataset):
    """
    Adapter for datasets that return dictionaries (like {'image': ..., 'label': ...})
    to work with DINOv3's expected (image, target) tuple format.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        """
        Args:
            base_dataset: Your existing dataset that returns dicts
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            transforms: Combined transform (takes precedence)
        """
        self.base_dataset = base_dataset
        
        if transforms is not None:
            self.transforms = transforms
            self.transform = None
            self.target_transform = None
        else:
            self.transform = transform
            self.target_transform = target_transform
            self.transforms = None
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        sample = self.base_dataset[idx]
        
        # Extract image and target from dict
        if isinstance(sample, dict):
            image = sample.get('image')
            target = sample.get('label', 0)
        else:
            # Already a tuple
            image, target = sample
        
        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms((image, target))
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
        
        return image, target


# # ============================================================================
# # Integration with DINOv3's make_dataset function
# # ============================================================================

# def register_custom_dataset():
#     """
#     Call this function to register your custom dataset with DINOv3's data loading.
    
#     You need to patch the _parse_dataset_str function in dinov3.data.loaders.
#     """
#     import dinov3.data.loaders as loaders_module
    
#     # Save the original function
#     _original_parse = loaders_module._parse_dataset_str
    
#     def _parse_dataset_str_extended(dataset_str: str):
#         """Extended parser that handles custom datasets"""
#         tokens = dataset_str.split(":")
#         name = tokens[0]
        
#         # Check if it's a custom dataset
#         if name == "MICROUS_SSL":
#             kwargs = {}
#             for token in tokens[1:]:
#                 if "=" in token:
#                     key, value = token.split("=")
#                     assert key in ("root", "extra", "split")
#                     kwargs[key] = value
            
#             if "split" in kwargs:
#                 kwargs["split"] = MedicalDataset.Split.__dict__[kwargs["split"]]
            
#             return MedicalDataset, kwargs
        
#         # Fall back to original parser for built-in datasets
#         return _original_parse(dataset_str)
    
#     # Monkey patch the function
#     loaders_module._parse_dataset_str = _parse_dataset_str_extended
#     logger.info("Custom dataset registered with DINOv3 data loading")


# # ============================================================================
# # Alternative: Direct usage without patching
# # ============================================================================

# def create_custom_dataset_for_dinov3(
#     root: str,
#     split: str = "train",
#     transform: Optional[Callable] = None,
#     target_transform: Optional[Callable] = None,
#     transforms: Optional[Callable] = None,
#     base_dataset: Optional[Dataset] = None,
#     **kwargs,
# ) -> Dataset:
#     """
#     Factory function to create a dataset compatible with DINOv3's training pipeline.
    
#     Use this if you don't want to register a custom dataset string parser.
    
#     Args:
#         root: Root directory of dataset
#         split: Dataset split
#         transform: Transform for images
#         target_transform: Transform for targets
#         transforms: Combined transform
#         base_dataset: If you have an existing dataset, wrap it with adapter
#         **kwargs: Additional arguments
        
#     Returns:
#         Dataset compatible with DINOv3
        
#     Example:
#         >>> # Option 1: Use CustomMedicalDataset directly
#         >>> dataset = create_custom_dataset_for_dinov3(
#         ...     root="/path/to/data",
#         ...     split="train",
#         ...     transform=my_transform
#         ... )
        
#         >>> # Option 2: Wrap existing dict-based dataset
#         >>> my_dataset = MyExistingDataset(...)
#         >>> dataset = create_custom_dataset_for_dinov3(
#         ...     root="",  # not used when base_dataset provided
#         ...     base_dataset=my_dataset,
#         ...     transform=my_transform
#         ... )
#     """
#     if base_dataset is not None:
#         # Wrap existing dataset
#         return DictDatasetAdapter(
#             base_dataset=base_dataset,
#             transform=transform,
#             target_transform=target_transform,
#             transforms=transforms,
#         )
#     else:
#         # Create new dataset
#         return MedicalDataset(
#             root=root,
#             split=MedicalDataset.Split.__dict__.get(split.upper(), split),
#             transform=transform,
#             target_transform=target_transform,
#             transforms=transforms,
#             **kwargs,
#         )


# # ============================================================================
# # Usage Examples
# # ============================================================================

# """
# # Example 1: Register custom dataset and use with dataset string
# register_custom_dataset()

# # Then in your config or train file, use:
# dataset_path = "CustomMedical:root=/path/to/data:split=TRAIN"
# dataset = make_dataset(
#     dataset_str=dataset_path,
#     transform=model.build_data_augmentation_dino(cfg),
#     target_transform=lambda _: (),
# )

# # Example 2: Direct usage without registration (modify build_data_loader_from_cfg)
# def build_data_loader_from_cfg_custom(cfg, model, start_iter):
#     # ... (collate_fn setup same as before)
    
#     # Instead of make_dataset, create directly:
#     dataset = create_custom_dataset_for_dinov3(
#         root=cfg.train.dataset_path,
#         split="train",
#         transform=model.build_data_augmentation_dino(cfg),
#         target_transform=lambda _: (),
#     )
    
#     # Or wrap your existing dataset:
#     # my_existing_dataset = get_dataset(**cfg.data)
#     # dataset = create_custom_dataset_for_dinov3(
#     #     root="",
#     #     base_dataset=my_existing_dataset,
#     #     transform=model.build_data_augmentation_dino(cfg),
#     # )
    
#     # Rest of the function remains the same
#     data_loader = make_data_loader(
#         dataset=dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         shuffle=True,
#         seed=cfg.train.seed + start_iter + 1,
#         sampler_type=sampler_type,
#         sampler_advance=start_iter * dataloader_batch_size_per_gpu,
#         drop_last=True,
#         collate_fn=collate_fn,
#     )
#     return data_loader

# # Example 3: If you have existing dataset that returns dicts
# from your_module import YourExistingDataset

# your_dataset = YourExistingDataset(root="/path", transform=None)
# wrapped_dataset = DictDatasetAdapter(
#     base_dataset=your_dataset,
#     transform=model.build_data_augmentation_dino(cfg),
#     target_transform=lambda _: (),
# )
# """