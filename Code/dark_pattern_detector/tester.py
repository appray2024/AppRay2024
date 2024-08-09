import os
from glob import glob
import json
from tqdm import tqdm
import numpy as np
import cv2
import pickle as pk
import datetime
import click

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

from transformers import AdamW

from src.utils import *
from src.models import *
from src.dataset import *

# Constants
DEVICE = "cuda:1"
N_EPOCH = 300
N_BATCH = 32
N_LAYERS = 50

OUR_DATASET_ROOT = "xx/dark_pattern/our_labelling_v2/round_4"
OUR_DATASET_IMGS = f"{OUR_DATASET_ROOT}/images"
OUR_DATASET_DETECTION = f"{OUR_DATASET_ROOT}/detection"
OUR_DATASET_IMG_TEXT = f"{OUR_DATASET_DETECTION}/text"
OUR_DATASET_PROC_DATA = f"{OUR_DATASET_DETECTION}/proc_dataset.pk"
OUR_DATASET_PROC_DATA_APP = f"{OUR_DATASET_DETECTION}/proc_dataset_app.pk"


def setup_logging():
    date_str = datetime.datetime.now().strftime("%d-%m-%Y")
    return get_global_logger(
        "DP_trainer",
        f"xx/dark_pattern/test_log_{date_str}.out",
    )


def get_model_checkpoint_filename(
    use_class_weight,
    use_negative_sampling,
    use_balance_augmentation,
    dl_model,
    base_path,
):
    filename = f"{base_path}/{N_BATCH}_best_ckpt_2"
    if use_class_weight:
        filename += "_cw"
    if use_negative_sampling:
        filename += "_ne"
    if use_balance_augmentation:
        filename += "_os"
    filename += f"_{dl_model}"
    return filename


def load_model(dl_model, total_labels, device):
    model = None
    tokenizer = None
    lr = 0.0
    model_type = ""

    if dl_model == "RESNET":
        model = SiameseResNet(
            n_channels=3, n_class=len(total_labels), n_layers=N_LAYERS
        )
        lr = 0.003
        model_type = "SiameseResNet"
    elif dl_model == "BERT":
        model = Bert_Classifier(n_class=len(total_labels))
        tokenizer = model.tokenizer
        lr = 3e-5
        model_type = "Bert_Classifier"
    elif dl_model in ["BERT-RESNET-F", "BERT-RESNET-NF"]:
        model = Bert_ResNet(n_channels=3, n_class=len(total_labels), n_layers=N_LAYERS)
        tokenizer = model.tokenizer
        model_type = "Bert_ResNet"

    return model, tokenizer, lr, model_type


def configure_optimizer(model, lr):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)


@click.command()
@click.option("--use-class-weight", is_flag=True, help="Enable class weight.")
@click.option("--use-negative-sampling", is_flag=True, help="Enable negative sampling.")
@click.option(
    "--use-balance-augmentation", is_flag=True, help="Enable balanced augmentation."
)
@click.option(
    "--dl-model",
    type=click.Choice(["RESNET", "BERT", "BERT-RESNET-F", "BERT-RESNET-NF"]),
    default="BERT-RESNET-F",
    help="Choose a model type.",
)
@click.option("--gpu", default=1, type=int, help="Choose a GPU.")
@click.option(
    "--output-pred", default="pred_floats.pk", type=str, help="Output prediction file."
)
def main(
    use_class_weight,
    use_negative_sampling,
    use_balance_augmentation,
    dl_model,
    gpu,
    output_pred,
):
    DEVICE = f"cuda:{gpu}"
    file_logger = setup_logging()

    # Load datasets
    our_train_set, our_test_set, our_labels, max_seq_length = load_dataset(
        OUR_DATASET_IMG_TEXT, OUR_DATASET_PROC_DATA_APP, OUR_DATASET_IMGS
    )
    total_labels = sorted(set(our_labels + [label_map[l] for l in rico_labels]))

    # Load model
    checkpoint_base_path = get_model_checkpoint_filename(
        use_class_weight,
        use_negative_sampling,
        use_balance_augmentation,
        dl_model,
        MODEL_OUTPUT_PREFIX,
    )
    model, tokenizer, lr, model_type = load_model(dl_model, total_labels, DEVICE)

    if model_type:
        if dl_model in ["BERT-RESNET-F", "BERT-RESNET-NF"]:
            f_resnet_encoder = get_pt_path(
                f"{checkpoint_base_path}_SiameseResNet_lr{r_lr}", "RESNET"
            )
            f_bert_encoder = get_pt_path(
                f"{checkpoint_base_path}_Bert_Classifier_lr{b_lr}", "BERT"
            )
            model.load_encoders(f_bert_encoder, f_resnet_encoder, DEVICE)
            if dl_model == "BERT-RESNET-F":
                model.freeze_encoders()

        checkpoint_path = get_pt_path(checkpoint_base_path, dl_model)
        model.load_state_dicts(checkpoint_path, DEVICE)
        model.to(DEVICE)

    # Set up optimizer
    optimizer = configure_optimizer(model, lr)

    # Prepare datasets and dataloaders
    our_test_set = one_hot_vector(our_test_set, total_labels)
    test_dataset = CustomDataset(all_test_set, tokenizer, max_seq_length)
    test_dataloader = DataLoader(test_dataset, batch_size=N_BATCH, shuffle=False)

    # Inference
    test_acc, test_loss, mac, mic, conf_mat, err_imgs, pred_floats = inference(
        model, test_dataloader, DEVICE, None
    )

    # Log results
    test_acc = round(test_acc, 4)
    file_logger.info(f"Test accuracy: {test_acc}")
    file_logger.info(f"Macro: {mac}")
    file_logger.info(f"Micro: {mic}")
    file_logger.info(f"Confusion Matrix: \n{conf_mat}")

    with open(f"{MODEL_OUTPUT_PREFIX}/err_msg.json", "w") as f:
        json.dump(err_imgs, f)
    with open(f"{MODEL_OUTPUT_PREFIX}/{output_pred}", "wb") as f:
        pk.dump(pred_floats, f)


if __name__ == "__main__":
    main()
