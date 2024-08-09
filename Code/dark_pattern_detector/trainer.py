import os
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

from src.loss import ContrastiveLoss

RICO_DATASET_ROOT = "xxx/dark_pattern/rico_testset"
RICO_DATASET_DETECTION = os.path.join(RICO_DATASET_ROOT, "detection")
RICO_DATASET_IMG_TEXT = os.path.join(RICO_DATASET_DETECTION, "text")
RICO_DATASET_PROC_DATA = os.path.join(RICO_DATASET_DETECTION, "proc_dataset.pk")

OUR_DATASET_ROOT = "xxx/dark_pattern/our_labelling_v2/round_4"
OUR_DATASET_IMGS = os.path.join(OUR_DATASET_ROOT, "images")
OUR_DATASET_DETECTION = os.path.join(OUR_DATASET_ROOT, "ui_group")
OUR_DATASET_IMG_TEXT = os.path.join(OUR_DATASET_DETECTION, "group_properties")
OUR_DATASET_PROC_DATA = os.path.join(OUR_DATASET_DETECTION, "group_proc_dataset.pk")
OUR_DATASET_PROC_DATA_APP = os.path.join(
    OUR_DATASET_DETECTION, "group_proc_dataset_app.pk"
)

VAL_DATASET_ROOT = f"{OUR_DATASET_ROOT}/ui_group/light_data/AppRay_Light_871"
VAL_DATASET_IMGS = os.path.join(VAL_DATASET_ROOT, "test")
VAL_DATASET_DETECTION = os.path.join(VAL_DATASET_ROOT, "ui_group")
VAL_DATASET_IMG_TEXT = os.path.join(VAL_DATASET_DETECTION, "test/group_properties")
VAL_DATASET_PROC_DATA = os.path.join(
    VAL_DATASET_DETECTION, "test/group_proc_dataset.pk"
)
VAL_DATASET_PROC_DATA_APP = os.path.join(
    VAL_DATASET_DETECTION, "test/group_proc_dataset_app.pk"
)

US_DATASET_ROOT = f"{OUR_DATASET_ROOT}/ui_group/user_study"
US_DATASET_IMGS = os.path.join(US_DATASET_ROOT, "imgToBeExamined")
US_DATASET_DETECTION = os.path.join(US_DATASET_ROOT, "ui_group")
US_DATASET_IMG_TEXT = os.path.join(US_DATASET_DETECTION, "group_properties")
US_DATASET_PROC_DATA = os.path.join(US_DATASET_DETECTION, "group_proc_dataset.pk")
US_DATASET_PROC_DATA_APP = os.path.join(
    US_DATASET_DETECTION, "group_proc_dataset_app.pk"
)

N_EPOCH = 200
N_BATCH = 32
N_LAYERS = 50
DEVICE = "cuda:1"
MODEL_OUTPUT_PREFIX = f"{OUR_DATASET_ROOT}/ui_group"
date_str = datetime.datetime.now().strftime("%d-%m-%Y")


def get_checkpoint_filename(
    checkpoint_base,
    use_class_weight,
    use_negative_sampling,
    use_balance_augmentation,
    model_type,
    lr,
):
    if use_class_weight:
        checkpoint_base += "_cw"
    if use_negative_sampling:
        checkpoint_base += "_ne"
    if use_balance_augmentation:
        checkpoint_base += "_os"
    return f"{checkpoint_base}_{model_type}_lr{lr}"


def load_model(dl_model, total_labels, device):
    if dl_model == "RESNET":
        model = SiameseResNet(
            n_channels=3, n_class=len(total_labels), n_layers=N_LAYERS
        )
        lr = 0.003
        model_type = "SiameseResNet"
    elif dl_model == "BERT":
        model = Bert_Classifier(n_class=len(total_labels))
        lr = 3e-5
        model_type = "Bert_Classifier"
    elif dl_model in ["BERT-RESNET-F", "BERT-RESNET-NF"]:
        model = Bert_ResNet(n_channels=3, n_class=len(total_labels), n_layers=N_LAYERS)
        r_lr = 0.003
        b_lr = 3e-5
        model_type = "Bert_ResNet"
        return model, r_lr, b_lr, model_type
    else:
        raise ValueError(f"Unsupported dl_model: {dl_model}")

    model.to(device)
    return model, lr, model_type


@click.command()
@click.option("--use-class-weight", default=False, help="Enable class weight.")
@click.option(
    "--use-negative-sampling", default=False, help="Enable negative sampling."
)
@click.option(
    "--use-balance-augmentation", default=False, help="Enable balanced augmentation."
)
@click.option(
    "--dl-model",
    type=click.Choice(["RESNET", "BERT", "BERT-RESNET-F", "BERT-RESNET-NF"]),
    default="BERT-RESNET-F",
    help="Choose a model type.",
)
@click.option("--gpu", default=1, type=int, help="Choose a GPU.")
@click.option(
    "--output-pred", default="pred_floats.pk", type=str, help="Output predict file."
)
def main(
    use_class_weight,
    use_negative_sampling,
    use_balance_augmentation,
    dl_model,
    gpu,
    output_pred,
):
    device = f"cuda:{gpu}"
    checkpoint_base = os.path.join(MODEL_OUTPUT_PREFIX, str(N_BATCH) + "_best_ckpt_3")
    checkpoint_file = get_checkpoint_filename(
        checkpoint_base,
        use_class_weight,
        use_negative_sampling,
        use_balance_augmentation,
        dl_model,
        0,
    )

    rico_train_set, rico_test_set, rico_labels, max_seq_length_1 = load_dataset(
        RICO_DATASET_IMG_TEXT, RICO_DATASET_PROC_DATA, RICO_DATASET_ROOT
    )
    our_train_set, our_test_set, our_labels, max_seq_length_2 = load_dataset(
        OUR_DATASET_IMG_TEXT, OUR_DATASET_PROC_DATA, OUR_DATASET_IMGS
    )
    _, our_us_set, _, _ = load_dataset(
        US_DATASET_IMG_TEXT, US_DATASET_PROC_DATA, US_DATASET_IMGS, test_only=True
    )

    max_seq_length = max(max_seq_length_1, max_seq_length_2)
    total_labels = sorted(set(our_labels + [label_map[l] for l in rico_labels]))

    tokenizer = None
    model, lr, model_type = load_model(dl_model, total_labels, device)

    if dl_model in ["BERT-RESNET-F", "BERT-RESNET-NF"]:
        r_lr, b_lr = model
        f_resnet_encoder = get_pt_path(
            f"{checkpoint_base}_SiameseResNet_lr{r_lr}", "RESNET"
        )
        f_bert_encoder = get_pt_path(
            f"{checkpoint_base}_Bert_Classifier_lr{b_lr}", "BERT"
        )
        model.load_encoders(f_bert_encoder, f_resnet_encoder, device)

    model_checkpoint_path = get_pt_path(checkpoint_file, dl_model)
    model.load_state_dicts(model_checkpoint_path, device)

    file_logger = get_global_logger(
        "DP_trainer",
        os.path.join(
            "xxx/dark_pattern/log_",
            f"{model_type}_{date_str}.out",
        ),
    )

    optimizer = AdamW(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "bias" not in n and "LayerNorm.weight" not in n
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "bias" in n or "LayerNorm.weight" in n
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
        eps=1e-8,
    )

    loss_fn = (
        ContrastiveLoss(weight=torch.Tensor(train_dataset.class_weight).to(device))
        if use_class_weight
        else ContrastiveLoss()
    )

    rico_train_set = one_hot_vector(rico_train_set, total_labels)
    rico_test_set = one_hot_vector(rico_test_set, total_labels)
    our_train_set = one_hot_vector(our_train_set, total_labels)
    our_test_set = one_hot_vector(our_test_set, total_labels)
    our_us_set = one_hot_vector(our_us_set, total_labels)

    all_train_set = rico_train_set + our_train_set + rico_test_set
    all_test_set = our_test_set

    train_dataset = CustomDataset(
        all_train_set, tokenizer, max_seq_length, over_sample=use_balance_augmentation
    )
    test_dataset = CustomDataset(all_test_set, tokenizer, max_seq_length)
    us_dataset = CustomDataset(our_us_set, tokenizer, max_seq_length)

    train_dataloader = DataLoader(train_dataset, batch_size=N_BATCH, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=N_BATCH, shuffle=False)
    us_dataloader = DataLoader(us_dataset, batch_size=N_BATCH, shuffle=False)

    file_logger.info(f"Checkpoint path: {checkpoint_file}")

    test_accs, test_macs, test_mics = [0.1], [[0.1, 0.1, 0.1]], [[0.1, 0.1, 0.1]]
    train_losses, test_f1s = [], []

    for epoch in tqdm(range(N_EPOCH)):
        model.train()
        train_loss = 0

        for batch_idx, (images_texts, labels, t_idx) in enumerate(train_dataloader):
            images_texts = [img_text.to(device) for img_text in images_texts]
            labels = labels.to(device)

            model.zero_grad()

            if isinstance(model, Bert_Classifier):
                output = model(images_texts, labels)
                loss = output["loss"]
            elif isinstance(model, Bert_ResNet):
                output, xs, bert_output = model(images_texts, labels)
                loss = loss_fn(xs, output, labels, negloss=use_negative_sampling)
                loss += bert_output["loss"]
            else:
                output, xs = model(images_texts, labels)
                loss = loss_fn(xs, output, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        train_acc, _, mac, mic, _, _, _ = inference(
            model, train_dataloader, device, loss_fn
        )
        file_logger.info(
            f"Epoch {epoch}, Train Loss: {avg_train_loss}, Train acc: {round(train_acc, 4)}, {mac}"
        )

        test_acc, test_loss, mac, mic, conf_mat, err_imgs, pred_floats = inference(
            model, test_dataloader, device, loss_fn
        )
        test_acc = round(test_acc, 4)
        test_accs.append(test_acc)
        test_macs.append(mac)
        test_mics.append(mic)
        test_f1s.append(mac[-1])

        p_max_test = np.argmax(test_accs)
        file_logger.info(
            f"Epoch {epoch}, Test acc: {test_acc}/{max(test_accs)}, test loss: {test_loss}"
        )
        file_logger.info(
            f"Epoch {epoch}, macro: {test_macs[p_max_test]}/{mac}, micro: {test_mics[p_max_test]}/{mic}"
        )

        us_pred_outputs = inference(model, us_dataloader, device, loss_fn)
        us_pred_floats = us_pred_outputs[-1]

        if mac[-1] == max(test_f1s):
            best_model = model
            best_model.save_state_dicts(checkpoint_file)
            file_logger.info(f"Confusion Matrix: \n{conf_mat}")

            with open(os.path.join(MODEL_OUTPUT_PREFIX, "err_msg.json"), "w") as f:
                json.dump(err_imgs, f)
            with open(os.path.join(MODEL_OUTPUT_PREFIX, output_pred), "wb") as f:
                pk.dump(pred_floats, f)
            with open(
                os.path.join(MODEL_OUTPUT_PREFIX, f"us_{output_pred}"), "wb"
            ) as f:
                pk.dump(us_pred_floats, f)


if __name__ == "__main__":
    main()
