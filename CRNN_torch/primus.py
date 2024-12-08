import cv2
import numpy as np
import ctc_utils
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import random
import pandas as pd


class CTC_PriMuS:
    """Handle CTC (Connectionist Temporal Classification) for PriMuS dataset."""

    GT_ELEMENT_SEPARATOR = "-"
    PAD_COLUMN = 0

    def __init__(
        self,
        corpus_dirpath: str,
        corpus_filepath: str,
        dictionary_path: str,
        semantic: bool,
        distortions: bool = False,
        val_split: float = 0.0,
    ) -> None:
        """
        Initialize the CTCPriMuS handler.

        Args:
            corpus_dirpath: Directory path containing the corpus
            corpus_filepath: Path to the corpus file
            dictionary_path: Path to the dictionary file
            semantic: Whether to use semantic annotations
            distortions: Whether to use distorted images
            val_split: Validation split ratio (0.0 to 1.0)
        """
        self.semantic = semantic
        self.distortions = distortions
        self.corpus_dirpath = Path(corpus_dirpath)
        self.validation_dict: Optional[Dict] = None
        self.current_idx = 0

        # Load corpus
        self.corpus_df = pd.read_csv(corpus_filepath)
        self.corpus_df["png"] = self.corpus_df["png"].apply(lambda x: "../" + x)

        corpus_list = self.corpus_df["file"].to_list()
        # with open(corpus_filepath, "r", encoding="utf-8") as corpus_file:
            # corpus_list = corpus_file.read().splitlines()

        # Load dictionary
        self.word2int = {}
        self.int2word = {}

        with open(dictionary_path, "r", encoding="utf-8") as dict_file:
            for word in dict_file.read().splitlines():
                if word not in self.word2int:
                    word_idx = len(self.word2int)
                    self.word2int[word] = word_idx
                    self.int2word[word_idx] = word

        self.vocabulary_size = len(self.word2int)

        # Split into training and validation sets
        random.shuffle(corpus_list)
        val_idx = int(len(corpus_list) * val_split)
        self.training_list = corpus_list[val_idx:]
        self.validation_list = corpus_list[:val_idx]

        print(
            f"Training with {len(self.training_list)} and validating with {len(self.validation_list)}"
        )

    def _load_image(self, sample_fullpath: Path) -> np.ndarray:
        """Load and prepare image for processing."""
        if self.distortions:
            img_path = f"{sample_fullpath}_distorted.jpg"
        else:
            img_path = f"{sample_fullpath}.png"

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        return img

    def _calculate_width_reduction(self, params: Dict) -> int:
        """
        Calculate width reduction based on convolution parameters.

        Handles both cases where conv_pooling_size can be:
        - A list of tuples [(h1,w1), (h2,w2), ...]
        - A list of integers [w1, w2, ...]
        """
        width_reduction = 1
        pooling_sizes = params["conv_pooling_size"]

        for i in range(params["conv_blocks"]):
            # Handle both tuple and integer cases
            if isinstance(pooling_sizes[i], (tuple, list)):
                width_reduction *= pooling_sizes[i][1]  # Use second element of tuple
            else:
                width_reduction *= pooling_sizes[i]  # Use the integer directly

        return width_reduction

    def _prepare_batch_images(
        self, images: List[np.ndarray], params: Dict
    ) -> np.ndarray:
        """Prepare batch of images with consistent dimensions."""
        max_width = max(img.shape[1] for img in images)
        # Rearrange dimensions to match PyTorch's expected NCHW format
        batch_shape = [
            len(images),
            params["img_channels"],  # Channels dimension moved to position 1
            params["img_height"],
            max_width,
        ]
        batch_images = np.ones(shape=batch_shape, dtype=np.float32) * self.PAD_COLUMN
        for i, img in enumerate(images):
            # Reshape the image to add channel dimension and transpose
            img_reshaped = img[np.newaxis, :, :]  # Add channel dimension
            batch_images[i, :, :img.shape[0], :img.shape[1]] = img_reshaped
        return batch_images

    def nextBatch(self, params: Dict) -> Dict:
        """Get next batch of training data."""
        images = []
        labels = []

        for _ in range(params["batch_size"]):
            sample_fullpath = self.corpus_df["png"][self.current_idx]
            # sample_filepath = self.training_list[self.current_idx]
            # sample_fullpath = self.corpus_dirpath / sample_filepath / sample_filepath

            # Load and process image
            sample_img = self._load_image(sample_fullpath)
            sample_img = ctc_utils.resize(sample_img, params["img_height"])
            images.append(ctc_utils.normalize(sample_img))

            # Load ground truth
            extension = ".semantic" if self.semantic else ".agnostic"
            gt_path = f"{sample_fullpath}{extension}"

            with open(gt_path, "r", encoding="utf-8") as gt_file:
                sample_gt_plain = (
                    gt_file.readline().rstrip().split(ctc_utils.word_separator())
                )
                labels.append([self.word2int[lab] for lab in sample_gt_plain])

            self.current_idx = (self.current_idx + 1) % len(self.training_list)

        # Prepare batch data
        batch_images = self._prepare_batch_images(images, params)
        width_reduction = self._calculate_width_reduction(params)
        lengths = [batch_images.shape[2] / width_reduction] * len(images)

        return {
            "inputs": batch_images,
            "seq_lengths": np.asarray(lengths),
            "targets": labels,
        }

    def getValidation(self, params: Dict) -> Tuple[Dict, int]:
        """Get validation dataset."""
        if self.validation_dict is None:
            images = []
            labels = []

            for sample_filepath in self.validation_list:
                sample_fullpath = self.corpus_df[self.corpus_df["file"] == sample_filepath]["png"].iloc[0]
                # sample_fullpath = (
                #     self.corpus_dirpath / sample_filepath / sample_filepath
                # )

                # Load and process image
                sample_img = self._load_image(sample_fullpath)
                sample_img = ctc_utils.resize(sample_img, params["img_height"])
                images.append(ctc_utils.normalize(sample_img))

                # Load ground truth
                extension = ".semantic" if self.semantic else ".agnostic"
                gt_path = f"{sample_fullpath}{extension}"

                with open(gt_path, "r", encoding="utf-8") as gt_file:
                    sample_gt_plain = (
                        gt_file.readline().rstrip().split(ctc_utils.word_separator())
                    )
                    labels.append([self.word2int[lab] for lab in sample_gt_plain])

            # Prepare validation data
            batch_images = self._prepare_batch_images(images, params)
            width_reduction = self._calculate_width_reduction(params)
            lengths = [batch_images.shape[2] / width_reduction] * len(images)

            self.validation_dict = {
                "inputs": batch_images,
                "seq_lengths": np.asarray(lengths),
                "targets": labels,
            }

        return self.validation_dict, len(self.validation_list)

    def getValidationSize(self) -> int:
        """Return the total size of validation dataset."""
        return len(self.validation_list)

    def getValidationBatch(self, start_idx: int, batch_size: int, params: Dict) -> Dict:
        """Get a batch of validation data."""
        end_idx = min(start_idx + batch_size, len(self.validation_list))
        batch_filepaths = self.validation_list[start_idx:end_idx]
        
        images = []
        labels = []
        
        for sample_filepath in batch_filepaths:
            # Get PNG path from DataFrame
            sample_fullpath = self.corpus_df[self.corpus_df["file"] == sample_filepath]["png"].iloc[0]
            
            # Load and process image
            sample_img = self._load_image(sample_fullpath)
            sample_img = ctc_utils.resize(sample_img, params["img_height"])
            images.append(ctc_utils.normalize(sample_img))
            
            # Load ground truth
            extension = ".semantic" if self.semantic else ".agnostic"
            gt_path = f"{sample_fullpath}{extension}"
            with open(gt_path, "r", encoding="utf-8") as gt_file:
                sample_gt_plain = gt_file.readline().rstrip().split(ctc_utils.word_separator())
                labels.append([self.word2int[lab] for lab in sample_gt_plain])
        
        # Prepare batch data
        batch_images = self._prepare_batch_images(images, params)
        width_reduction = self._calculate_width_reduction(params)
        lengths = [batch_images.shape[2] / width_reduction] * len(images)
        
        return {
            "inputs": batch_images,
            "seq_lengths": np.asarray(lengths),
            "targets": labels,
        }
