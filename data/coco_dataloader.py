import os
import json
import random
from collections import defaultdict
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class COCODataset(Dataset):    
    def __init__(
        self, 
        image_dir: str, 
        annotations_file: str, 
        image_size: int = 64,
        transform: Optional[transforms.Compose] = None,
        max_caption_length: Optional[int] = None
    ):
        self.image_dir = os.path.expanduser(image_dir)
        self.annotations_file = os.path.expanduser(annotations_file)
        self.image_size = image_size
        self.max_caption_length = max_caption_length
        
        # Load annotations
        with open(self.annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images']}
        
        # Group captions by image_id
        self.id_to_captions = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.id_to_captions[ann['image_id']].append(ann['caption'])
        
        # Filter out images that don't have captions or don't exist
        self.image_ids = []
        for img_id in self.id_to_filename.keys():
            if img_id in self.id_to_captions:
                image_path = os.path.join(self.image_dir, self.id_to_filename[img_id])
                if os.path.exists(image_path):
                    self.image_ids.append(img_id)
        
        print(f"Found {len(self.image_ids)} valid images with captions")
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            image: Tensor of shape (3, image_size, image_size)
            caption: String caption for the image
        """
        img_id = self.image_ids[idx]
        
        # Load and transform image
        image_path = os.path.join(self.image_dir, self.id_to_filename[img_id])
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # Get a random caption for this image
        captions = self.id_to_captions[img_id]
        caption = random.choice(captions)
        
        # Truncate caption if necessary
        if self.max_caption_length is not None:
            caption = caption[:self.max_caption_length]
        
        return image, caption
    
    def get_all_captions_for_image(self, idx: int) -> List[str]:
        """Get all captions for an image at given index."""
        img_id = self.image_ids[idx]
        return self.id_to_captions[img_id]


def create_coco_dataloader(
    image_dir: str,
    annotations_file: str,
    batch_size: int = 16,
    image_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Optional[transforms.Compose] = None,
    max_caption_length: Optional[int] = None
) -> DataLoader:
    dataset = COCODataset(
        image_dir=image_dir,
        annotations_file=annotations_file,
        image_size=image_size,
        transform=transform,
        max_caption_length=max_caption_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )


def collate_fn(batch: List[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
    images, captions = zip(*batch)
    images = torch.stack(images)
    return images, list(captions)


def get_coco_train_dataloader(
    image_dir: str = "~/Downloads/coco2017/train2017",
    annotations_file: str = "~/Downloads/coco2017/annotations/captions_train2017.json",
    batch_size: int = 16,
    image_size: int = 64
) -> DataLoader:
    return create_coco_dataloader(
        image_dir=image_dir,
        annotations_file=annotations_file,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )


def get_coco_val_dataloader(
    image_dir: str = "~/Downloads/coco2017/val2017",
    annotations_file: str = "~/Downloads/coco2017/annotations/captions_val2017.json",
    batch_size: int = 16,
    image_size: int = 64
) -> DataLoader:
    return create_coco_dataloader(
        image_dir=image_dir,
        annotations_file=annotations_file,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
