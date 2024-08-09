import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Type

from torch.nn import BCEWithLogitsLoss

import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import pickle as pk


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, stride=1
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.batch_norm3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_channels, planes, downsample, stride)]
        self.in_channels = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)


class CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CNN, self).__init__()
        self.cnn = ResNet(
            Bottleneck, layers=[2, 2, 2, 2], num_channels=n_channels
        )  # Assuming ResNet-18
        self.fc = nn.Linear(512 * 2, n_classes)

    def forward(self, x):
        sub_img, full_img = x
        x_s = self.cnn(sub_img)
        x_f = self.cnn(full_img)
        x = torch.hstack((x_s, x_f))
        return self.fc(x)


class Bert_ResNet(nn.Module):
    def __init__(
        self, n_channels, n_classes, n_layers=50, bert_name="bert-base-uncased"
    ):
        super(Bert_ResNet, self).__init__()
        self.bert_name = bert_name
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.text_classifier = AutoModelForSequenceClassification.from_pretrained(
            bert_name, num_labels=n_classes
        )
        self.text_encoder = self.text_classifier.bert
        self.dropout = nn.Dropout(0.1)
        self.save_encoder = True

        self.resnet_cnn = ResNet(
            Bottleneck, layers=[3, 4, 6, 3], num_channels=n_channels
        )  # Assuming ResNet-50

        self.fc = nn.Sequential(nn.Linear(512 * 2 + 768, 64), nn.Linear(64, n_classes))

    def load_encoders(self, f_bert, f_resnet, device="cuda:0"):
        if os.path.exists(f_bert):
            bert_state_dict = torch.load(f_bert, map_location=torch.device(device))
            self.text_classifier.load_state_dict(bert_state_dict["model_state_dict"])
        else:
            raise ValueError(f"BERT checkpoint {f_bert} not found.")

        if os.path.exists(f_resnet):
            resnet_state_dict = torch.load(f_resnet, map_location=torch.device(device))
            self.resnet_cnn.load_state_dict(resnet_state_dict["model_state_dict"])
        else:
            raise ValueError(f"ResNet checkpoint {f_resnet} not found.")

    def load_state_dicts(self, f_ckpt, device="cuda:0"):
        if os.path.exists(f_ckpt):
            ckpt_state_dict = torch.load(f_ckpt, map_location=torch.device(device))
            if "bert_encoder" in ckpt_state_dict:
                self.text_classifier.load_state_dict(ckpt_state_dict["bert_encoder"])
            if "resnet_encoder" in ckpt_state_dict:
                self.resnet_cnn.load_state_dict(ckpt_state_dict["resnet_encoder"])

            self.fc.load_state_dict(ckpt_state_dict["model_state_dict"])
        else:
            print(f"ERROR: Checkpoint {f_ckpt} not found. No checkpoint loaded.")

    def save_state_dicts(self, f_ckpt):
        model_ckpt = {"model_state_dict": self.fc.state_dict()}

        if self.save_encoder:
            model_ckpt["bert_encoder"] = self.text_classifier.state_dict()
            model_ckpt["resnet_encoder"] = self.resnet_cnn.state_dict()

        torch.save(model_ckpt, f_ckpt)

    def freeze_encoders(self, n_encoder="all"):
        if n_encoder in ["ResNet", "all"]:
            for param in self.resnet_cnn.parameters():
                param.requires_grad = False

        if n_encoder in ["bert", "all"]:
            for param in self.text_classifier.parameters():
                param.requires_grad = False

        self.save_encoder = False

    def forward(self, x, y):
        sub_img, full_img, input_ids, attention_mask, token_type_ids = x
        t_y = self.text_classifier(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=y,
        )
        t_emb = self.text_encoder(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )[1]
        t_emb = self.dropout(t_emb)

        x_s = self.resnet_cnn(sub_img)
        x_f = self.resnet_cnn(full_img)
        x = torch.hstack((x_s, x_f, t_emb))
        x = self.dropout(x)

        return self.fc(x), x, t_y


class Bert_Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Bert_Classifier, self).__init__()
        self.bert_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name)
        self.text_classifier = AutoModelForSequenceClassification.from_pretrained(
            self.bert_name, num_labels=n_classes, hidden_dropout_prob=0.5
        )

    def load_state_dicts(self, f_model, device="cuda:0"):
        if os.path.exists(f_model):
            bert_state_dict = torch.load(f_model, map_location=torch.device(device))
            self.text_classifier.load_state_dict(bert_state_dict["model_state_dict"])
        else:
            print(
                f"ERROR: BERT classifier checkpoint {f_model} not found. No checkpoint loaded."
            )

    def save_state_dicts(self, f_bert):
        print(f"Saving BERT classifier state dict to {f_bert}")
        torch.save({"model_state_dict": self.text_classifier.state_dict()}, f_bert)

    def forward(self, x, y):
        if len(x) > 2:
            _, _, input_ids, attention_mask, token_type_ids = x
            t_y = self.text_classifier(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=y,
            )
        else:
            input_ids, attention_mask = x
            t_y = self.text_classifier(
                input_ids, attention_mask=attention_mask, labels=y
            )

        return t_y


class SiameseResNet(nn.Module):
    def __init__(self, n_class: int, n_layers: int = 50, n_channels: int = 3):
        super(SiameseResNet, self).__init__()

        self.resnet_cnn = ResNet(
            num_channels=n_channels, num_layers=n_layers, ResBlock=Bottleneck,
        )
        self.fc = nn.Linear(512 * 2, n_class)

    def save_state_dicts(self, f_resnet):
        torch.save(
            {
                "model_state_dict": self.resnet_cnn.state_dict(),
                "fc_state_dict": self.fc.state_dict(),
            },
            f_resnet,
        )

    def load_state_dicts(self, f_resnet, device="cuda:0"):
        if os.path.exists(f_resnet):
            resnet_state_dict = torch.load(f_resnet, map_location=torch.device(device))
            self.resnet_cnn.load_state_dict(resnet_state_dict["model_state_dict"])
            self.fc.load_state_dict(resnet_state_dict["fc_state_dict"])
        else:
            print(f"ERROR: {f_resnet} checkpoint not existed. No checkpoint loaded.")

    def forward(self, x, y):
        sub_img, full_img = x
        x_s = self.resnet_cnn(sub_img)
        x_f = self.resnet_cnn(full_img)
        x = torch.hstack((x_s, x_f))
        return self.fc(x), x
