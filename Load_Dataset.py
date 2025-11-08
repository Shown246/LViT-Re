# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
from transformers import BertModel, BertTokenizer
import numpy as np
import pandas as pd
from getEm import get_embeddings

# class BertEmbeddingWrapper:
#     def __init__(self, model_name='microsoft/BiomedVLP-CXR-BERT-specialized', use_cuda=True):
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = BertModel.from_pretrained(model_name)
#         self.model.eval()

#         # device logic
#         if use_cuda and torch.cuda.is_available():
#             self.device = 'cuda'
#         else:
#             self.device = 'cpu'
            
#         # self.device = 'cpu'
#         print(f"BertEmbeddingWrapper using device: {self.device}")
#         self.model.to(self.device)

#     def __call__(self, sentences):
#         results = []
#         with torch.no_grad():
#             # batch tokenize
#             inputs = self.tokenizer(
#                 sentences,
#                 return_tensors='pt',
#                 padding=True,
#                 truncation=True,
#                 max_length=512
#             ).to(self.device)

#             outputs = self.model(**inputs)
#             last_hidden = outputs.last_hidden_state          # [B, T, D]

#             # loop per item to match your old return structure
#             for i in range(last_hidden.shape[0]):
#                 token_ids = inputs['input_ids'][i].cpu()
#                 tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

#                 token_embeddings = last_hidden[i].cpu().numpy()
#                 results.append((tokens, token_embeddings))

#         return results

def random_rot_flip(image, label):
    """Apply random rotation (90° increments) and random flip"""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    """Apply random rotation within ±20 degrees"""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    """Data augmentation transform for training with random rotations/flips"""
    
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        
        # Apply random augmentation (50% chance each)
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # Resize if dimensions don't match
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # Convert to tensors
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


class ValGenerator(object):
    """Validation transform without augmentation, only resizing"""
    
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        
        # Resize if needed (no augmentation for validation)
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # Convert to tensors
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


def to_long_tensor(pic):
    """Convert numpy array to long tensor"""
    img = torch.from_numpy(np.array(pic, np.uint8))
    return img.long()


def correct_dims(*images):
    """Add channel dimension if image is 2D (grayscale)"""
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class LV2D(Dataset):
    """Dataset for label-only data (no images) with text descriptions"""
    
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        # self.bert_embedding = BertEmbeddingWrapper()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.output_path))

    def __getitem__(self, idx):
        mask_filename = self.mask_list[idx]
        mask_path = os.path.join(self.output_path, mask_filename)
        
        # Load and preprocess mask
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
            
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        mask = correct_dims(mask)
        
        # Get text description and generate BERT embeddings
        text = self.rowtext[mask_filename]
        text = text.split('\n')
        text_token = self.bert_embedding(text)
        text = np.array(text_token[0][1])
        
        # Truncate text embeddings to max 14 tokens
        if text.shape[0] > 14:
            text = text[:14, :]
            
        # Convert mask to one-hot encoding if specified
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'label': mask, 'text': text}

        return sample, mask_filename


class ImageToImage2D(Dataset):
    """Dataset for image segmentation with images, masks, and text descriptions"""

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        # self.bert_embedding = BertEmbeddingWrapper()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        # Get filenames based on dataset type
        if self.task_name == 'Covid19':
            # Covid19: use index for both image and mask lists
            image_filename = self.images_list[idx]
            mask_filename = self.mask_list[idx]
        else:  # MoNuSeg
            # MoNuSeg: derive mask filename from image filename
            image_filename = self.images_list[idx]
            mask_filename = image_filename[: -3] + "png"

        # print(f"Processing image: {image_filename}, mask: {mask_filename}")
        text_token = get_embeddings(mask_filename)
        # Build full paths
        image_path = os.path.join(self.input_path, image_filename)
        mask_path = os.path.join(self.output_path, mask_filename)

        # Load and resize image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Load and resize mask
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        
        # Binarize mask
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # Ensure correct dimensions
        image, mask = correct_dims(image, mask)

        # Get text description and generate BERT embeddings
        text = self.rowtext[mask_filename]
        text = text.split('\n')
        # text_token = self.bert_embedding(text)
        text = np.array(text_token[0][1])
        
        # Truncate text embeddings to max 10 tokens
        # In your data loading or preprocessing
        # print(f"Text shape: {text.shape}")  # Should be [batch, 10, seq_len]
        if text.shape[0] > 10:
            text = text[:10, :]

        # Convert mask to one-hot encoding if specified
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'image': image, 'label': mask, 'text': text}

        # Apply transforms (augmentation/normalization)
        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename