"""
Dataset utilities for GLM-Image finetuning.

This module provides dataset loading and preprocessing for:
- Text-to-Image (T2I) training using text-to-image-2M dataset
- Image-to-Image (I2I) training using HQ-Edit dataset

Datasets:
- T2I: jackyhate/text-to-image-2M
  - data_1024_10K: 10K high-quality 1024x1024 images with prompts (Flux-dev generated)
  - data_512_2M: 2M 512x512 images for larger-scale training

- I2I: UCSC-VLAA/HQ-Edit
  - 197K+ high-quality instruction-based image editing pairs
  - GPT-4V + DALL-E 3 generated dataset
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset

    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Run: pip install datasets")


# Default data directory for downloaded datasets
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""

    # Dataset source (auto-selected based on task_type if not specified)
    dataset_name: str = ""  # Will be set based on task_type
    subset: str = "data_1024_10K"  # T2I: "data_1024_10K" or "data_512_2M"

    # Image processing
    resolution: int = 1024
    center_crop: bool = True
    random_flip: bool = True

    # Data loading
    streaming: bool = False  # HQ-Edit doesn't support streaming well
    cache_dir: Optional[str] = None  # If None, uses DEFAULT_DATA_DIR
    num_workers: int = 4
    prefetch_factor: int = 2

    # Training mode
    task_type: str = "t2i"  # "t2i" or "i2i"

    # Prompt processing
    max_prompt_length: int = 512
    add_prompt_prefix: str = ""  # Optional prefix for all prompts

    # I2I specific: use edit instruction or output description as prompt
    i2i_prompt_type: str = "edit"  # "edit" or "output_description"

    # For custom local datasets
    local_data_dir: Optional[str] = None
    metadata_file: str = "metadata.jsonl"

    def __post_init__(self):
        """Set up cache directory after initialization."""
        if self.cache_dir is None:
            self.cache_dir = DEFAULT_DATA_DIR

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)


class TextToImage2MDataset(Dataset):
    """
    Dataset wrapper for jackyhate/text-to-image-2M.

    This dataset contains high-quality text-image pairs suitable for
    finetuning text-to-image models.

    Args:
        config: DatasetConfig with dataset parameters
        transform: Optional image transform function
        tokenizer: Optional tokenizer for text processing
        split: Dataset split ("train" by default)
    """

    def __init__(
        self,
        config: DatasetConfig,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        split: str = "train",
    ):
        self.config = config
        self.transform = transform
        self.tokenizer = tokenizer
        self.split = split

        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "The 'datasets' library is required. "
                "Install it with: pip install datasets"
            )

        self._load_dataset()

    def _load_dataset(self):
        """Load the dataset from HuggingFace Hub."""
        if self.config.subset == "data_1024_10K":
            # 10K high-resolution dataset
            data_url = (
                "https://huggingface.co/datasets/jackyhate/text-to-image-2M/"
                "resolve/main/data_1024_10K/data_000000.tar"
            )
            self.dataset = load_dataset(
                "webdataset",
                data_files={"train": data_url},
                split=self.split,
                streaming=self.config.streaming,
                cache_dir=self.config.cache_dir,
            )
        elif self.config.subset == "data_512_2M":
            # 2M dataset - multiple shards
            num_shards = 46
            base_url = (
                "https://huggingface.co/datasets/jackyhate/text-to-image-2M/"
                "resolve/main/data_512_2M/data_{i:06d}.tar"
            )
            urls = [base_url.format(i=i) for i in range(num_shards)]
            self.dataset = load_dataset(
                "webdataset",
                data_files={"train": urls},
                split=self.split,
                streaming=self.config.streaming,
                cache_dir=self.config.cache_dir,
            )
        else:
            raise ValueError(f"Unknown subset: {self.config.subset}")

        # Convert to list if not streaming (for __len__ support)
        if not self.config.streaming:
            self.dataset = list(self.dataset)

    def __len__(self) -> int:
        if self.config.streaming:
            # For streaming dataset, return estimated size
            if self.config.subset == "data_1024_10K":
                return 10000
            elif self.config.subset == "data_512_2M":
                return 2000000
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.config.streaming:
            raise NotImplementedError(
                "Indexing not supported for streaming dataset. "
                "Use iterate() method or set streaming=False."
            )

        sample = self.dataset[idx]
        return self._process_sample(sample)

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample from the dataset."""
        # Extract image
        image = sample.get("jpg") or sample.get("png") or sample.get("image")
        if isinstance(image, bytes):
            import io

            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = self._default_transform(image)

        # Extract prompt from JSON
        json_data = sample.get("json", {})
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        prompt = json_data.get("prompt", "")

        # Add optional prefix
        if self.config.add_prompt_prefix:
            prompt = f"{self.config.add_prompt_prefix} {prompt}"

        result = {
            "image": image,
            "prompt": prompt,
        }

        # Tokenize if tokenizer provided
        if self.tokenizer:
            tokens = self.tokenizer(
                prompt,
                max_length=self.config.max_prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids"] = tokens["input_ids"].squeeze(0)
            result["attention_mask"] = tokens["attention_mask"].squeeze(0)

        return result

    def _default_transform(self, image: Image.Image) -> torch.Tensor:
        """Default image transformation."""
        import torchvision.transforms as T

        transforms = []

        # Resize
        if self.config.center_crop:
            transforms.extend(
                [
                    T.Resize(
                        self.config.resolution,
                        interpolation=T.InterpolationMode.BILINEAR,
                    ),
                    T.CenterCrop(self.config.resolution),
                ]
            )
        else:
            transforms.append(
                T.Resize(
                    (self.config.resolution, self.config.resolution),
                    interpolation=T.InterpolationMode.BILINEAR,
                )
            )

        # Random horizontal flip
        if self.config.random_flip:
            transforms.append(T.RandomHorizontalFlip(p=0.5))

        # To tensor and normalize
        transforms.extend(
            [
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
            ]
        )

        transform = T.Compose(transforms)
        return transform(image)

    def iterate(self):
        """
        Iterate over the dataset (works for both streaming and non-streaming).

        Yields:
            Processed samples as dictionaries.
        """
        if self.config.streaming:
            for sample in self.dataset:
                yield self._process_sample(sample)
        else:
            for idx in range(len(self)):
                yield self[idx]


class LocalT2IDataset(Dataset):
    """
    Dataset for local text-to-image data.

    Expected directory structure:
    ```
    data_dir/
    ├── images/
    │   ├── image001.jpg
    │   ├── image002.jpg
    │   └── ...
    └── metadata.jsonl
    ```

    metadata.jsonl format:
    ```
    {"image": "images/image001.jpg", "prompt": "A detailed description..."}
    ```
    """

    def __init__(
        self,
        data_dir: str,
        config: Optional[DatasetConfig] = None,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.data_dir = data_dir
        self.config = config or DatasetConfig()
        self.transform = transform
        self.tokenizer = tokenizer

        self.samples = self._load_metadata()

    def _load_metadata(self) -> List[Dict[str, str]]:
        """Load metadata from JSONL file."""
        metadata_path = os.path.join(self.data_dir, self.config.metadata_file)

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        samples = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load image
        image_path = os.path.join(self.data_dir, sample["image"])
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = self._default_transform(image)

        prompt = sample.get("prompt", "")
        if self.config.add_prompt_prefix:
            prompt = f"{self.config.add_prompt_prefix} {prompt}"

        result = {
            "image": image,
            "prompt": prompt,
        }

        if self.tokenizer:
            tokens = self.tokenizer(
                prompt,
                max_length=self.config.max_prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids"] = tokens["input_ids"].squeeze(0)
            result["attention_mask"] = tokens["attention_mask"].squeeze(0)

        return result

    def _default_transform(self, image: Image.Image) -> torch.Tensor:
        """Default image transformation."""
        import torchvision.transforms as T

        transforms = [
            T.Resize(
                self.config.resolution, interpolation=T.InterpolationMode.BILINEAR
            ),
            T.CenterCrop(self.config.resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]

        if self.config.random_flip:
            transforms.insert(2, T.RandomHorizontalFlip(p=0.5))

        return T.Compose(transforms)(image)


class LocalI2IDataset(Dataset):
    """
    Dataset for local image-to-image data.

    Expected directory structure:
    ```
    data_dir/
    ├── source_images/
    │   ├── source001.jpg
    │   └── ...
    ├── target_images/
    │   ├── target001.jpg
    │   └── ...
    └── metadata.jsonl
    ```

    metadata.jsonl format:
    ```
    {"source": "source_images/source001.jpg", "target": "target_images/target001.jpg", "prompt": "Edit instruction..."}
    ```
    """

    def __init__(
        self,
        data_dir: str,
        config: Optional[DatasetConfig] = None,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.data_dir = data_dir
        self.config = config or DatasetConfig(task_type="i2i")
        self.transform = transform
        self.tokenizer = tokenizer

        self.samples = self._load_metadata()

    def _load_metadata(self) -> List[Dict[str, str]]:
        """Load metadata from JSONL file."""
        metadata_path = os.path.join(self.data_dir, self.config.metadata_file)

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        samples = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load source and target images
        source_path = os.path.join(self.data_dir, sample["source"])
        target_path = os.path.join(self.data_dir, sample["target"])

        source_image = Image.open(source_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        # Apply transforms
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
        else:
            source_image = self._default_transform(source_image)
            target_image = self._default_transform(target_image)

        prompt = sample.get("prompt", "")
        if self.config.add_prompt_prefix:
            prompt = f"{self.config.add_prompt_prefix} {prompt}"

        result = {
            "source_image": source_image,
            "target_image": target_image,
            "prompt": prompt,
        }

        if self.tokenizer:
            tokens = self.tokenizer(
                prompt,
                max_length=self.config.max_prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids"] = tokens["input_ids"].squeeze(0)
            result["attention_mask"] = tokens["attention_mask"].squeeze(0)

        return result

    def _default_transform(self, image: Image.Image) -> torch.Tensor:
        """Default image transformation."""
        import torchvision.transforms as T

        transforms = [
            T.Resize(
                self.config.resolution, interpolation=T.InterpolationMode.BILINEAR
            ),
            T.CenterCrop(self.config.resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]

        return T.Compose(transforms)(image)


class HQEditDataset(Dataset):
    """
    Dataset wrapper for UCSC-VLAA/HQ-Edit.

    HQ-Edit is a high-quality instruction-based image editing dataset with 197K+ edits.
    Generated using GPT-4V and DALL-E 3.

    Dataset structure:
    - input_image: source image
    - output_image: target/edited image
    - edit: editing instruction (e.g., "Change the sky to sunset")
    - input: description of input image
    - output: description of output image
    - inverse_edit: reverse editing instruction

    Args:
        config: DatasetConfig with dataset parameters
        transform: Optional image transform function
        tokenizer: Optional tokenizer for text processing
        split: Dataset split ("train" by default)
    """

    def __init__(
        self,
        config: DatasetConfig,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        split: str = "train",
    ):
        self.config = config
        self.transform = transform
        self.tokenizer = tokenizer
        self.split = split

        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "The 'datasets' library is required. "
                "Install it with: pip install datasets"
            )

        self._load_dataset()

    def _load_dataset(self):
        """Load the HQ-Edit dataset from HuggingFace Hub."""
        self.dataset = load_dataset(
            "UCSC-VLAA/HQ-Edit",
            split=self.split,
            cache_dir=self.config.cache_dir,
        )

        print(f"Loaded HQ-Edit dataset with {len(self.dataset)} samples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        return self._process_sample(sample)

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample from the dataset."""
        # Extract images
        source_image = sample["input_image"]
        target_image = sample["output_image"]

        # Ensure PIL Image format
        if not isinstance(source_image, Image.Image):
            source_image = Image.open(source_image).convert("RGB")
        else:
            source_image = source_image.convert("RGB")

        if not isinstance(target_image, Image.Image):
            target_image = Image.open(target_image).convert("RGB")
        else:
            target_image = target_image.convert("RGB")

        # Apply transforms
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
        else:
            source_image = self._default_transform(source_image)
            target_image = self._default_transform(target_image)

        # Select prompt based on config
        if self.config.i2i_prompt_type == "edit":
            # Use editing instruction: "Change the sky to sunset"
            prompt = sample.get("edit", "")
        elif self.config.i2i_prompt_type == "output_description":
            # Use output description: "A landscape with a beautiful sunset sky"
            prompt = sample.get("output", "")
        else:
            prompt = sample.get("edit", "")

        # Add optional prefix
        if self.config.add_prompt_prefix:
            prompt = f"{self.config.add_prompt_prefix} {prompt}"

        result = {
            "source_image": source_image,
            "target_image": target_image,  # This becomes "image" for training
            "image": target_image,  # Alias for compatibility with T2I training flow
            "prompt": prompt,
            # Additional metadata
            "input_description": sample.get("input", ""),
            "output_description": sample.get("output", ""),
            "edit_instruction": sample.get("edit", ""),
            "inverse_edit": sample.get("inverse_edit", ""),
        }

        # Tokenize if tokenizer provided
        if self.tokenizer:
            tokens = self.tokenizer(
                prompt,
                max_length=self.config.max_prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids"] = tokens["input_ids"].squeeze(0)
            result["attention_mask"] = tokens["attention_mask"].squeeze(0)

        return result

    def _default_transform(self, image: Image.Image) -> torch.Tensor:
        """Default image transformation."""
        import torchvision.transforms as T

        transforms_list = []

        # Resize
        if self.config.center_crop:
            transforms_list.extend(
                [
                    T.Resize(
                        self.config.resolution,
                        interpolation=T.InterpolationMode.BILINEAR,
                    ),
                    T.CenterCrop(self.config.resolution),
                ]
            )
        else:
            transforms_list.append(
                T.Resize(
                    (self.config.resolution, self.config.resolution),
                    interpolation=T.InterpolationMode.BILINEAR,
                )
            )

        # Note: No random flip for I2I to maintain source-target correspondence

        # To tensor and normalize
        transforms_list.extend(
            [
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
            ]
        )

        transform = T.Compose(transforms_list)
        return transform(image)

    def iterate(self):
        """
        Iterate over the dataset.

        Yields:
            Processed samples as dictionaries.
        """
        for idx in range(len(self)):
            yield self[idx]


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with optimized settings for training.

    Args:
        dataset: The dataset to load from
        batch_size: Batch size (limited by GPU memory for GLM-Image)
        shuffle: Whether to shuffle the data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch

    Returns:
        Configured DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching samples.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with stacked tensors
    """
    result = {}

    # Stack images
    if "image" in batch[0]:
        result["image"] = torch.stack([sample["image"] for sample in batch])
    if "source_image" in batch[0]:
        result["source_image"] = torch.stack(
            [sample["source_image"] for sample in batch]
        )
    if "target_image" in batch[0]:
        result["target_image"] = torch.stack(
            [sample["target_image"] for sample in batch]
        )

    # Collect prompts as list
    result["prompt"] = [sample["prompt"] for sample in batch]

    # Stack tokenized inputs if present
    if "input_ids" in batch[0]:
        result["input_ids"] = torch.stack([sample["input_ids"] for sample in batch])
    if "attention_mask" in batch[0]:
        result["attention_mask"] = torch.stack(
            [sample["attention_mask"] for sample in batch]
        )

    return result


# Map friendly names to HuggingFace dataset IDs and column names
HF_DATASET_MAPPING = {
    "pokemon": {
        "id": "reach-vb/pokemon-blip-captions",
        "image_col": "image",
        "text_col": "text",
        "resolution": 512,
    },
}


class HuggingFaceImageTextDataset(Dataset):
    """
    Generic wrapper for HuggingFace image-text datasets.
    """

    def __init__(
        self,
        config: DatasetConfig,
        dataset_id: str,
        image_col: str = "image",
        text_col: str = "text",
        default_prompt: str = "",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        split: str = "train",
    ):
        self.config = config
        self.transform = transform
        self.tokenizer = tokenizer
        self.image_col = image_col
        self.text_col = text_col
        self.default_prompt = default_prompt

        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "The 'datasets' library is required. "
                "Install it with: pip install datasets"
            )

        print(f"Loading HF dataset: {dataset_id}")
        self.dataset = load_dataset(
            dataset_id,
            split=split,
            cache_dir=self.config.cache_dir,
            verification_mode="no_checks",
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        max_retries = 20
        current_idx = idx
        sample = None
        image = None

        for attempt in range(max_retries):
            try:
                sample = self.dataset[current_idx]

                # Get image
                if self.image_col not in sample:
                    for col in ["image", "img", "jpg"]:
                        if col in sample:
                            self.image_col = col
                            break

                image = sample.get(self.image_col)

                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert("RGB")
                else:
                    image = image.convert("RGB")

                # If success, break the retry loop
                break
            except Exception as e:
                # If this was the last attempt, re-raise
                if attempt == max_retries - 1:
                    print(f"Error loading sample {current_idx} (final attempt): {e}")
                    raise RuntimeError(
                        f"Failed to load any valid samples after {max_retries} retries starting from index {idx}"
                    ) from e

                # Otherwise try next sample
                # print(f"Error loading sample {current_idx}: {e}, trying next...")
                current_idx = (current_idx + 1) % len(self)

        # Get prompt
        prompt = self.default_prompt
        if self.text_col in sample and sample[self.text_col]:
            prompt = str(sample[self.text_col])

        if self.config.add_prompt_prefix:
            prompt = f"{self.config.add_prompt_prefix} {prompt}"

        # Apply transforms
        if self.transform:
            pixel_values = self.transform(image)
        else:
            pixel_values = self._default_transform(image)

        result = {
            "image": pixel_values,
            "prompt": prompt,
        }

        if self.tokenizer:
            tokens = self.tokenizer(
                prompt,
                max_length=self.config.max_prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids"] = tokens["input_ids"].squeeze(0)
            result["attention_mask"] = tokens["attention_mask"].squeeze(0)

        return result

    def _default_transform(self, image: Image.Image) -> torch.Tensor:
        import torchvision.transforms as T

        transforms = [
            T.Resize(
                self.config.resolution, interpolation=T.InterpolationMode.BILINEAR
            ),
            T.CenterCrop(self.config.resolution)
            if self.config.center_crop
            else T.RandomCrop(self.config.resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
        return T.Compose(transforms)(image)


def get_dataset(
    config: DatasetConfig,
    transform: Optional[Callable] = None,
    tokenizer: Optional[Any] = None,
) -> Dataset:
    """
    Factory function to get the appropriate dataset based on config.
    """
    if config.local_data_dir:
        # Use local dataset
        if config.task_type == "t2i":
            return LocalT2IDataset(
                data_dir=config.local_data_dir,
                config=config,
                transform=transform,
                tokenizer=tokenizer,
            )
        elif config.task_type == "i2i":
            return LocalI2IDataset(
                data_dir=config.local_data_dir,
                config=config,
                transform=transform,
                tokenizer=tokenizer,
            )
    else:
        # Check if subset is in HF_DATASET_MAPPING
        if config.subset in HF_DATASET_MAPPING:
            mapping = HF_DATASET_MAPPING[config.subset]
            return HuggingFaceImageTextDataset(
                config=config,
                dataset_id=mapping["id"],
                image_col=mapping.get("image_col", "image"),
                text_col=mapping.get("text_col", "text"),
                default_prompt=mapping.get("default_prompt", ""),
                transform=transform,
                tokenizer=tokenizer,
            )

        # Use HuggingFace dataset
        if config.task_type == "t2i":
            return TextToImage2MDataset(
                config=config,
                transform=transform,
                tokenizer=tokenizer,
            )
        elif config.task_type == "i2i":
            return HQEditDataset(
                config=config,
                transform=transform,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(f"Unknown task type: {config.task_type}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--subset", type=str, default="pokemon")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--task_type", type=str, default="t2i")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache downloaded datasets (default: finetune/data)",
    )
    args = parser.parse_args()

    if args.task_type == "t2i":
        print(f"Loading T2I dataset: {args.subset}")

        config = DatasetConfig(
            subset=args.subset,
            streaming=args.streaming,
            resolution=1024,
            task_type="t2i",
            cache_dir=args.cache_dir,
        )

        if args.subset in HF_DATASET_MAPPING:
            config.resolution = HF_DATASET_MAPPING[args.subset].get("resolution", 512)

        dataset = get_dataset(config)
        print(f"Dataset size: {len(dataset)}")

        for i, sample in enumerate(DataLoader(dataset, batch_size=1)):
            if i >= args.num_samples:
                break
            print(
                f"Sample {i}: shape={sample['image'].shape}, prompt={sample['prompt'][0]}"
            )
