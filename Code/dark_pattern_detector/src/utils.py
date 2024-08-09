import logging
import logging.handlers
import os
import sys
import warnings
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

from src.models import *

global_logger = None

label_map = {
    "II-PRE": "II-Preselection",
    "II-AM-FH": "II-AM-FalseHierarchy-BAD",
    "II-AM-DA": "II-AM-DisguisedAD",
    "FA-SOCIALPYRAMID": "ForcedAction-SocialPyramid",
    "FA-Privacy": "ForcedAction-Privacy",
    "NG": "Nagging",
    "II-AM-G-SMALL": "II-AM-General",
    "FA-G-PRO": "ForcedAction-General",
    "FA-G-COUNTDOWNAD": "ForcedAction-General",
    "FA-G-WATCHAD": "ForcedAction-General",
    "SN-FC": "Sneaking-ForcedContinuity",
}


class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            if line.endswith("\n"):
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf:
            self.logger.log(self.log_level, self.linebuf.rstrip())
            self.linebuf = ""


def _init_logger(logger_name, logger_filename):
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.basicConfig(level=logging.INFO, encoding="utf-8")
    logging.getLogger().handlers[0].setFormatter(formatter)

    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    handler = logging.handlers.TimedRotatingFileHandler(
        logger_filename, when="D", utc=True, encoding="utf-8"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_global_logger(logger_name, logger_filename):
    global global_logger
    if global_logger is None:
        global_logger = _init_logger(logger_name, logger_filename)
    return global_logger


def inference(model, data_loader, device, loss_fn=None):
    model.eval()
    all_loss = 0.0
    sigmoid = torch.nn.Sigmoid()

    data_idxs, pred_labels, gth_labels = [], [], []
    err_imgs = []
    pred_floats = []

    with torch.no_grad():
        for images, labels, idxs in data_loader:
            images = [img.to(device) for img in images]
            labels = labels.to(device)
            data_idxs.extend(idxs.cpu().tolist())

            if isinstance(model, Bert_Classifier):
                output = model(images, labels)["logits"]
            elif isinstance(model, Bert_ResNet):
                output, xs, bert_output = model(images, labels)
                if loss_fn:
                    all_loss += (
                        loss_fn(xs, output, labels).item() + bert_output["loss"].item()
                    )
            else:
                output, xs = model(images, labels)
                if loss_fn:
                    all_loss += loss_fn(xs, output, labels).item()

            output = sigmoid(output).cpu().numpy()
            gth_label = labels.cpu().numpy()

            for i, l_gth in enumerate(gth_label):
                w_gth = np.where(l_gth == 1)[0].tolist()
                y_bar = np.zeros(len(output[i]))
                l_ybar = np.argsort(output[i])[-len(w_gth) :]
                for l in l_ybar:
                    if output[i][l] > 0.5:
                        y_bar[l] = 1

                gth_labels.append(l_gth)
                pred_labels.append(y_bar)

                w_pred = np.where(y_bar == 1)[0].tolist()
                for w in w_pred:
                    if w not in w_gth:
                        err_imgs.append(
                            [idxs[i].item(), y_bar.tolist(), l_gth.tolist()]
                        )

                output[i][output[i] <= 0.5] = 0
                pred_floats.append([idxs[i].item(), output[i], l_gth])

    all_loss /= len(data_loader.dataset)
    acc = accuracy_score(gth_labels, pred_labels)

    ma_p, ma_r, ma_f1, _ = precision_recall_fscore_support(
        gth_labels, pred_labels, average="macro", zero_division=np.nan
    )
    mi_p, mi_r, mi_f1, _ = precision_recall_fscore_support(
        gth_labels, pred_labels, average="micro"
    )
    conf_matrix = confusion_matrix(
        [np.argmax(g) for g in gth_labels], [np.argmax(p) for p in pred_labels]
    )

    return (
        acc,
        all_loss,
        [round(ma_p, 4), round(ma_r, 4), round(ma_f1, 4)],
        [round(mi_p, 4), round(mi_r, 4), round(mi_f1, 4)],
        conf_matrix,
        err_imgs,
        pred_floats,
    )


def one_hot_vector(dataset, total_labels):
    for s in dataset:
        v = np.zeros(len(total_labels))
        for l in s[-1]:
            l = label_map.get(l, l)
            if l in total_labels:
                v[total_labels.index(l)] = 1
        s[-1] = v

        if np.sum(v) > 1:
            print(s[0])
    return dataset


def get_pt_path(root, model_type):
    model_paths = {
        "RESNET": f"{root}_resnet_only.pt",
        "BERT": f"{root}_bert_only.pt",
        "BERT-RESNET-F": f"{root}_bert_resnet.pt",
        "BERT-RESNET-NF": f"{root}_bert_resnet.pt",
    }
    if model_type not in model_paths:
        raise ValueError(f"Error: unknown model: {model_type}")
    return model_paths[model_type]
