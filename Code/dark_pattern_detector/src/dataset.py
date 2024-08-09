import os
import pandas as pd
from collections import Counter
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import nlpaug.flow as naf
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

import cv2
from glob import glob
from tqdm import tqdm
import pickle as pk
import json
import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.utils import *

DP_JSON_FILE = "*_group.json"
class_threshold = 5


class CustomDataset(Dataset):
    def __init__(self, data_set, tokenizer=None, max_seq_length=0, over_sample=False):
        # Define transformations for sub-images and full images
        self.transform_subimg = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((133, 100), antialias=False),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        self.transform_fullimg = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((640, 480), antialias=False),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        # Define text augmentation transformations
        self.transform_text = naw.Sequential(
            [
                naw.RandomWordAug("delete"),
                naw.RandomWordAug("swap"),
                nas.RandomSentAug(),
            ]
        )

        self.data_set = data_set
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self._prepare_dataset(over_sample)
        self._prepare_labels()

        self.sub_imgs = []
        self.full_imgs = []
        self.token_rets = []
        self.one_hot_labels = []

        self._process_samples(over_sample)

        # Compute class weights for handling class imbalance
        self.class_weight = compute_class_weight(
            class_weight="balanced", classes=np.unique(self.labels), y=self.labels
        )

        # Assertions to ensure data consistency
        assert len(self.one_hot_labels) == len(self.sub_imgs) == len(self.full_imgs)
        if self.tokenizer:
            assert len(self.one_hot_labels) == len(self.token_rets)

    def _prepare_dataset(self, over_sample):
        """Prepare the dataset by oversampling if required."""
        self.clear_data_set = []
        self.others = []

        if over_sample:
            for ds in self.data_set:
                one_hot_label = ds[-1]
                if one_hot_label[0] == 0:
                    self.others.append(ds)
                else:
                    self.clear_data_set.append(ds)

            max_count = 603
            if len(self.clear_data_set) > 603:
                sampled_clear_set = random.sample(self.clear_data_set, max_count)
            else:
                sampled_clear_set = self.clear_data_set

            self.sampled_dataset = sampled_clear_set + self.others
        else:
            self.sampled_dataset = self.data_set

    def _prepare_labels(self):
        """Extract and count labels from the sampled dataset."""
        self.labels = [np.argmax(ds[-1]) for ds in self.sampled_dataset]
        self.labels_counts = Counter(self.labels)

    def _process_samples(self, over_sample):
        """Process each sample in the dataset and apply transformations."""
        for ds in self.sampled_dataset:
            img_id, sub_img, bnd, f_full_img, text, all_text, one_hot_label = ds
            full_img_cv_ = cv2.imread(f_full_img)

            n_loop = self._calculate_loop_count(over_sample, one_hot_label)

            for _ in range(n_loop):
                self._add_sample(sub_img, full_img_cv_, text, all_text, one_hot_label)

    def _calculate_loop_count(self, over_sample, one_hot_label):
        """Calculate the number of times to loop for oversampling."""
        if over_sample:
            l_count = self.labels_counts[np.argmax(one_hot_label)]
            return max(1, round((603 - l_count) / l_count))
        return 1

    def _add_sample(self, sub_img, full_img_cv_, text, all_text, one_hot_label):
        """Add a processed sample to the dataset."""
        self.sub_imgs.append(self.transform_subimg(sub_img))

        try:
            full_img_cv = self.transform_fullimg(full_img_cv_)
        except Exception:
            full_img_cv = torch.full((3, 640, 480), 0.00001)

        self.full_imgs.append(full_img_cv)

        if self.tokenizer:
            text = self.transform_text.augment(text)[0]
            token_ret = self.tokenizer(
                ", ".join([text, all_text]),
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
                return_token_type_ids=True,
                return_attention_mask=True,
            )
            self.token_rets.append(token_ret)

        self.one_hot_labels.append(one_hot_label)

    def __len__(self):
        return len(self.one_hot_labels)

    def __getitem__(self, idx):
        sub_img = self.sub_imgs[idx]
        full_img_cv = self.full_imgs[idx]
        one_hot_label = self.one_hot_labels[idx]

        if self.token_rets:
            input_ids = self.token_rets[idx]["input_ids"]
            attention_mask = self.token_rets[idx]["attention_mask"]
            token_type_ids = self.token_rets[idx]["token_type_ids"]

            return (
                [
                    sub_img,
                    full_img_cv,
                    np.array(input_ids),
                    np.array(attention_mask),
                    np.array(token_type_ids),
                ],
                one_hot_label,
                idx,
            )

        return [sub_img, full_img_cv], one_hot_label, idx


def prepare_dataset(path_img_text):
    """Prepares the dataset by loading and organizing image and text data."""

    # Load image-text paths and initialize dictionaries for storing data
    json_paths = glob(f"{path_img_text}/{DP_JSON_FILE}")
    img_ids = [f"{os.path.basename(p).split('_')[0]}.png" for p in json_paths]

    img_all_texts = {}
    class_img_bnd_text = defaultdict(list)

    # Parse the JSON files to extract image-text data
    for json_path in tqdm(json_paths, desc="Processing JSON files"):
        with open(json_path, "r") as f:
            group_props = json.load(f)
            img_id = group_props["image"]
            ori_img_path = group_props["ori_img_path"]
            text_file = f"{path_img_text}/{img_id}_all_text_ph.json"

            # Load all associated texts for the image
            if os.path.exists(text_file):
                with open(text_file, "r") as f:
                    all_texts = json.load(f)
                    if all_texts:
                        img_all_texts[img_id] = " ".join(
                            [t[0] for t in all_texts if len(t) == 2]
                        )

            # Process bounding boxes and labels
            for prop in group_props["properties"]:
                bnd, labels, text = (
                    prop["bbox"],
                    prop["labels"],
                    " ".join(prop["text"]).strip(),
                )

                labels = [
                    "II-HiddenInformation"
                    if l in ["II-HiddenInformation-TEXT", "II-HiddenInformation-ICON"]
                    else l
                    for l in labels
                    if l != "II-HiddenInformation-ICON"
                ]

                for label in labels:
                    class_img_bnd_text[label].append(
                        [img_id, ori_img_path, bnd, text, labels]
                    )

    # Handle specific label cases by merging or deleting entries
    if "NG" in class_img_bnd_text:
        class_img_bnd_text["NG"] = (
            class_img_bnd_text.pop("NG-UPGRADE", [])
            + class_img_bnd_text.pop("NG-RATE", [])
            + class_img_bnd_text.pop("NG-AD", [])
        )
    if "II-PRE" in class_img_bnd_text:
        class_img_bnd_text["II-PRE"] += class_img_bnd_text.pop("II-PRE-Nocheckbox", [])

    class_img_bnd_text = {
        k: v
        for k, v in class_img_bnd_text.items()
        if k not in {"NG-UPGRADE", "NG-RATE", "NG-AD", "Include"}
    }

    # Normalize labels within the dataset
    for _, img_bnd_texts in class_img_bnd_text.items():
        for img_bnd_text in img_bnd_texts:
            img_bnd_text[-1] = [
                "II-PRE" if "II-PRE" in lbl else "NG" if "NG" in lbl else lbl
                for lbl in img_bnd_text[-1]
            ]

    labels = list(class_img_bnd_text.keys())

    return class_img_bnd_text, img_all_texts, labels


def load_dataset_pre_class(f_in_img_text, f_out_proc_data, f_root, test_only=False):
    """Loads and prepares the dataset based on a class-based approach."""

    if not os.path.exists(f_out_proc_data):
        class_img_bnd_text, img_all_texts, _ = prepare_dataset(f_in_img_text)

        train_img_bnd_text = defaultdict(list)
        test_img_bnd_text = defaultdict(list)
        max_seq_length = 0

        for label, img_bnd_texts in class_img_bnd_text.items():
            n_sample = (
                round(len(img_bnd_texts) * 0.8)
                if len(img_bnd_texts) >= class_threshold
                else max(len(img_bnd_texts) - 1, 1)
            )

            if not test_only:
                train_img_bnd_text[label] = img_bnd_texts[:n_sample]
                test_img_bnd_text[label] = img_bnd_texts[n_sample:]
            else:
                test_img_bnd_text[label] = img_bnd_texts

        train_set, test_set = [], []
        for label, img_bnd_texts in train_img_bnd_text.items():
            for img_bnd_text in img_bnd_texts:
                train_set.append(
                    process_img_text(
                        img_bnd_text, img_all_texts, f_root, max_seq_length
                    )
                )

        for label, img_bnd_texts in test_img_bnd_text.items():
            for img_bnd_text in img_bnd_texts:
                test_set.append(
                    process_img_text(
                        img_bnd_text, img_all_texts, f_root, max_seq_length
                    )
                )

        with open(f_out_proc_data, "wb") as f:
            pk.dump(
                [train_set, test_set, list(train_img_bnd_text.keys()), max_seq_length],
                f,
            )
    else:
        with open(f_out_proc_data, "rb") as f:
            train_set, test_set, labels, max_seq_length = pk.load(f)

    return train_set, test_set, labels, max_seq_length


def load_dataset_pre_app(f_in_img_text, f_out_proc_data, f_root, test_only=False):
    """Loads and prepares the dataset based on an application-based approach."""

    if not os.path.exists(f_out_proc_data):
        class_img_bnd_text, img_all_texts, _ = prepare_dataset(f_in_img_text)

        class_app_counts = defaultdict(list)
        for _, img_bnd_texts in class_img_bnd_text.items():
            for img_name, _, _, _, labels in img_bnd_texts:
                app_name = img_name.split("_")[0]
                for label in labels:
                    if label != "Obstruction":
                        class_app_counts[label].append(app_name)

        class_app_counts = {k: list(set(v)) for k, v in class_app_counts.items()}
        class_app_counts = {
            k: len(v)
            for k, v in sorted(class_app_counts.items(), key=lambda item: item[1])
        }

        train_apps, test_apps = split_apps(class_app_counts)

        train_set, test_set = [], []
        for label, img_bnd_texts in class_img_bnd_text.items():
            for img_bnd_text in img_bnd_texts:
                train_set, test_set = process_app_data(
                    img_bnd_text,
                    img_all_texts,
                    f_root,
                    max_seq_length,
                    train_apps,
                    test_apps,
                    train_set,
                    test_set,
                )

        with open(f_out_proc_data, "wb") as f:
            pk.dump(
                [train_set, test_set, list(class_img_bnd_text.keys()), max_seq_length],
                f,
            )
    else:
        with open(f_out_proc_data, "rb") as f:
            train_set, test_set, labels, max_seq_length = pk.load(f)

    return train_set, test_set, labels, max_seq_length


def load_dataset(f_in_img_text, f_out_proc_data, f_root, test_only=False):
    """Determines the appropriate loading method based on the output file name."""

    if "proc_dataset_app" not in f_out_proc_data:
        return load_dataset_pre_class(f_in_img_text, f_out_proc_data, f_root, test_only)
    else:
        return load_dataset_pre_app(f_in_img_text, f_out_proc_data, f_root, test_only)


def process_img_text(img_bnd_text, img_all_texts, f_root, max_seq_length):
    """Processes image and text data, updating the maximum sequence length if necessary."""

    img_id, ori_img_path, bnd, text, y_gths = img_bnd_text
    f_full_img = f"{f_root}/{img_id}.jpg" if img_id[-4:] != ".png" else ori_img_path

    sub_img = get_sub_image(bnd, f_full_img)
    all_text = img_all_texts.get(img_id, "")
    text = "[This element does not contain texts]" if not text else text

    max_seq_length = max(max_seq_length, len(text.split(" ")))

    return [img_id, sub_img, bnd, f_full_img, text, all_text, y_gths]


def process_app_data(
    img_bnd_text,
    img_all_texts,
    f_root,
    max_seq_length,
    train_apps,
    test_apps,
    train_set,
    test_set,
):
    """Processes application data for training and testing sets."""

    img_id, ori_img_path, bnd, text, y_gths = img_bnd_text
    app_name = img_id.split("_")[0]
    f_full_img = f"{f_root}/{img_id}.jpg" if img_id[-4:] != ".png" else ori_img_path

    sub_img = get_sub_image(bnd, f_full_img)
    all_text = img_all_texts.get(img_id, "")
    text = "[This element does not contain texts]" if not text else text

    max_seq_length = max(max_seq_length, len(text.split(" ")))

    if app_name in train_apps:
        train_set.append([img_id, sub_img, bnd, f_full_img, text, all_text, y_gths])
    else:
        test_set.append([img_id, sub_img, bnd, f_full_img, text, all_text, y_gths])

    return train_set, test_set


def split_apps(class_app_counts):
    """Splits the applications into training and testing sets based on the class counts."""

    train_apps, test_apps = [], []
    for label, n in class_app_counts.items():
        app_names = class_app_counts[label]

        nsample = 1 if n < 5 else 2 if n < 10 else int(len(app_names) * 0.1)

        test_samples = app_names[:nsample]
        train_samples = app_names[nsample:]

        train_apps.extend([t for t in train_samples if t not in test_apps])
        test_apps.extend([t for t in test_samples if t not in train_apps])

    return train_apps, test_apps


def get_sub_image(sub_img_bnd, img_path):
    """Extracts a sub-image from the full image based on the bounding box."""

    full_img_cv = cv2.imread(img_path)
    mask = np.zeros_like(full_img_cv[:, :, 0])
    pts = np.array(sub_img_bnd, dtype=np.int32)

    cv2.fillPoly(mask, [pts], 255)
    masked_image = cv2.bitwise_and(full_img_cv, full_img_cv, mask=mask)
    x, y, w, h = cv2.boundingRect(pts)
    sub_image = masked_image[y : y + h, x : x + w]

    return sub_image
