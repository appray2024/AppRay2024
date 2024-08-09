import json
import numpy as np
import pickle as pk
import re
import sys
import click

from PIL import Image
import cv2
from glob import glob
from sentence_transformers import SentenceTransformer, util

from src.utils import *
from src.models import *
from src.dataset import *

from shapely.geometry import Polygon
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score


OUR_DATASET_ROOT = "/data/scsgpu1/work/jeffwang/dark_pattern/our_labelling_v2/round_4"
OUR_DATASET_IMGS = f"{OUR_DATASET_ROOT}/images"
OUR_DATASET_DETECTION = f"{OUR_DATASET_ROOT}/ui_group"
OUR_DATASET_IMG_TEXT = f"{OUR_DATASET_DETECTION}/group_properties"
OUR_DATASET_PROC_DATA = f"{OUR_DATASET_DETECTION}/group_proc_dataset.pk"
OUR_DATASET_PROC_DATA_APP = f"{OUR_DATASET_DETECTION}/group_proc_dataset_app.pk"


def calculate_iou(bnd_det, bnd_prop, div="det"):
    poly_det = Polygon(bnd_det)
    poly_prop = Polygon(bnd_prop)

    try:
        intersection_area = poly_det.intersection(poly_prop).area
        if div == "det":
            return intersection_area / poly_det.area
        else:
            return intersection_area / poly_prop.area
    except ZeroDivisionError:
        return 0.0


def bnd_convert_points(bnd):
    return [[bnd[0], bnd[1]], [bnd[2], bnd[1]], [bnd[2], bnd[3]], [bnd[0], bnd[3]]]


def point_convert_rect(bnd):
    return {
        "left": bnd[0][0],
        "top": bnd[0][1],
        "right": int(bnd[2][0]),
        "bottom": int(bnd[2][1]),
    }


def bnd_to_width_height(bnd):
    width = int(bnd[1][0]) - int(bnd[0][0])
    height = int(bnd[-1][1]) - int(bnd[0][1])
    return {"width": width, "height": height}


def find_check_group(bnd, text, all_properties):
    results = []

    for i, property in enumerate(all_properties):
        if "bbox" not in property:
            continue

        prop_bnd = bnd_convert_points(property["bbox"])
        text_tmp = property.get("text", "no text")
        iou = calculate_iou(bnd, prop_bnd)

        fg_lum = property.get("bg_lum")
        if fg_lum is None and "meta_items" in property:
            fg_lum = next(
                (
                    cg.get("bg_lum")
                    for cg in property["meta_items"]
                    if "text" in cg and cg["text"].lower() == text
                ),
                None,
            )

        results.append([i, text_tmp, iou, property["category"], prop_bnd, fg_lum])

    return results


def sort_filenames(filenames):
    def extract_key(filename):
        match = re.search(r"(T\d+|step)-?(\d+)", filename)
        if match:
            task = match.group(1)
            sequence = int(match.group(2))
            return (task, sequence)
        return (filename, 0)

    return sorted(filenames, key=extract_key)


def find_next_img(img_name, dict_imgs):
    app_name = img_name.split("_")[0]
    app_imgs = dict_imgs.get(app_name, [])
    try:
        f_idx = app_imgs.index(img_name)
        return app_imgs[f_idx + 1] if f_idx + 1 < len(app_imgs) else None
    except ValueError:
        return None


def find_img_label(
    img_id, train_dataset, test_dataset, pred_test_labels, app_or_label_set="app"
):
    labels = []

    if app_or_label_set != "app":
        for ds in train_dataset.data_set:
            img_id_, _, _, _, _, _, one_hot_label = ds
            if img_id == img_id_:
                labels.extend(
                    [tatal_labels[i] for i, l in enumerate(one_hot_label) if l == 1]
                )
                return list(set(labels))

    for i, ds in enumerate(test_dataset.data_set):
        img_id_, _, _, _, _, _, _ = ds
        if img_id == img_id_:
            one_hot_label = pred_test_labels[i]
            labels.extend(
                [tatal_labels[i] for i, l in enumerate(one_hot_label) if l == 1]
            )
            break

    return list(set(labels))


def jaccard_similarity(set1, set2):
    set1 = set(set1)
    set2 = set(set2)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0


def rule_check(text, all_properties, target, bnd=None):
    """
    Evaluate text and bounding box against various rules based on the target category.

    Parameters:
        text (str): The text content to check.
        all_properties (list): List of all properties to evaluate against.
        target (str): The rule target category to match.
        bnd (list or None): Bounding box coordinates (optional).

    Returns:
        bool: True if the text and bounding box match the rules for the target category, otherwise False.
    """
    text = text.lower().replace("-", " ").replace(",", " ").strip()
    ret = False

    if bnd is not None:
        bnd = [[int(num) for num in sublist] for sublist in bnd]

    # Define patterns for different target categories
    patterns = {
        "II-AM-ToyingWithEmotion": (
            re.compile(
                r"^(.*)(limited|check\s?in|new|credit|seize|special|exclusive).*(time|days|free|now|card|discount|offer|gift).*(offer|gift)?(.*)$",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(1[0-2]|0?[1-9]):([0-5][0-9])(\s?[APMapm]{2})?\b|\b([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?\b"
            ),
        ),
        "Obstruction-Currency": (
            re.compile(
                r"^(?!.*%)(?!.*\b\d{1,2}:\d{2}\b)(.*)(\$|¥|£|€)?\s*\d+(\s+)?(value)?"
            ),
            ["coin", "coins", "dollar", "pound", "credits"],
            ["yearly", "monthly"],
        ),
        "ForcedAction-Privacy": (
            re.compile(
                r"^(.*)\b(acknowledge|consent|agree|accept|share|send|analyze|collect|gather|offer|personalized|learn|stored|allow|use)\b(.*)\b(privacy|cookies|terms|policy|agreement|data|advertising|info|content)\b(.*)$",
                re.IGNORECASE,
            ),
        ),
        "II-HiddenInformation": (
            re.compile(
                r"^(.*)(terms of use|collection statement|privacy policy|accept all cookies|policies|policy|google play account|terms and conditions|terms & conditions|terms of service|license agreement)(.*)"
            )
        ),
        "II-Preselection": (
            re.compile(r"^(.*)(change|save)(.+)(preferences)(.+)(setting)(.*)"),
        ),
        "Sneaking-ForcedContinuity": (
            re.compile(
                r"^(.*)(free|automatic|trial|subscription|try|offer)?.*(then|after|renew|charged|subscription|trial).*(hours|days|weeks|months|years|period|cancel|ends|turned\s?off)?.*",
                re.IGNORECASE,
            ),
            re.compile(r"^(.*)(\$|¥|£|€)\s*\d+(\/?).*(year|month)"),
        ),
        "II-AM-DisguisedAD": (
            re.compile(
                r"^(?!watch)(.*)(chat|chat now|\sad(\s|\.|$)+|\sads|shop now|promoted|sponsor|sponsored|sponsorship|show your support|free to download|downloading|shop now|install|learn more|advertisement|\.com|Ad:|open).*",
                re.IGNORECASE,
            ),
        ),
        "ForcedAction-General": (
            re.compile(
                r"^(?!.*YouTube Advertising OPEN AD)(?!.*Turn Off Faster Browsing With No Ads).*"
                r"(no|skip|remove|watch|ad).+(ads|after|advertising|ad|free|(?:([01]?\d|2[0-3]):([0-5]?\d):)?([0-5]?\d)).*",
                re.IGNORECASE,
            ),
        ),
        "Obstruction-RoachMotel": (
            re.compile(r"^(.*)(apply).+(delete|remove).+(account).*")
        ),
        "II-AM-FalseHierarchy-BAD": (),
        "Nagging": (),
        "II-AM-Tricked": (re.compile(r"^(.*)(do).+(not).*")),
        "ForcedAction-SocialPyramid": (
            re.compile(r"^(.*)(share|refer|get).+(friend).*")
        ),
        "ForcedAction-Gamification": (
            re.compile(
                r"^(.*)(complete|finish|maintain|achieve|collect|earn|reward|streak|congratulations).+(streak|reward|congratulations|collect|earn).*",
                re.IGNORECASE,
            ),
        ),
        "II-AM-General": (),
    }

    def check_pattern(patterns, key, text):
        pattern = patterns.get(key, [None])[0]
        return pattern.search(text) is not None if pattern else False

    def check_currency(text):
        symbols_pattern = patterns["Obstruction-Currency"][0]
        return symbols_pattern.search(text) is not None or any(
            term in text for term in patterns["Obstruction-Currency"][1]
        )

    def check_exclusive(text):
        return any(word in text for word in patterns["Obstruction-Currency"][2])

    def check_group(bnd, text, all_properties):
        return find_check_group(bnd, text, all_properties)

    if target == "II-AM-ToyingWithEmotion":
        if (
            check_pattern(patterns, "II-AM-ToyingWithEmotion", text)
            or text == "[this element does not contain texts]".lower()
        ):
            return True

    elif target == "Obstruction-Currency":
        if check_currency(text) and not check_exclusive(text):
            return True

    elif target == "ForcedAction-Privacy":
        return check_pattern(patterns, "ForcedAction-Privacy", text)

    elif target == "II-HiddenInformation":
        if check_pattern(patterns, "II-HiddenInformation", text):
            return True
        if target == "II-Preselection":
            preselect_pattern = patterns["II-Preselection"][0]
            if preselect_pattern.search(text):
                return True
            group_ret = []
            find_groups = check_group(bnd, text, all_properties)
            concat_texts = []
            concat_texts_scores = 0
            checkgroup_size = None

            for g_i, g_text, g_iou, g_category, g_prop_bnd, g_fg_lum in find_groups:
                if g_category == "check_group" and checkgroup_size is None:
                    checkgroup_size = g_prop_bnd
                if g_iou > 0:
                    if rule_check(text, all_properties, "II-AM-Tricked", bnd):
                        if g_category == "check_group":
                            if all_properties[g_i]["status"][0] == "checked":
                                group_ret.append(True)
                        elif g_category in [
                            "TextView",
                            "pText",
                            "Button",
                            "EditText",
                            "ImageView",
                        ]:
                            concat_texts.append(
                                g_text.lower().replace(",", " ").strip()
                            )
                            concat_texts_scores += g_iou
            if concat_texts and concat_texts_scores > 0.06:
                concat_texts = " ".join(concat_texts)
                e1 = text_ranker.encode(text, convert_to_tensor=True)
                e2 = text_ranker.encode(concat_texts, convert_to_tensor=True)
                score = util.pytorch_cos_sim(e1, e2).tolist()[0][0]
                if score > 0.50:
                    group_ret.append(True)
                if checkgroup_size is not None:
                    hierarchy_elements = [
                        Polygon(checkgroup_size).area,
                        Polygon(bnd).area,
                    ]
                    if round(max(hierarchy_elements) / min(hierarchy_elements)) >= 2:
                        group_ret.append(False)
            return any(group_ret)

    elif target == "Sneaking-ForcedContinuity":
        if check_pattern(patterns, "Sneaking-ForcedContinuity", text) or re.compile(
            r"^(.*)(\$|¥|£|€)\s*\d+(\/?).*(year|month)"
        ).search(text):
            find_groups = check_group(bnd, text, all_properties)
            for g_i, g_text, g_iou, g_category, g_prop_bnd, g_fg_lum in find_groups:
                if g_iou > 0 and g_category == "ImageView":
                    return False
            return True

    elif target == "II-AM-DisguisedAD":
        if bnd[-1][-1] - bnd[0][-1] < 200:
            return False
        disguised_ad_pattern = patterns["II-AM-DisguisedAD"][0]
        if disguised_ad_pattern.search(text):
            rect = bnb_to_width_height(bnd)
            if rect["width"] > 1000 and rect["height"] > 2000:
                return False
        concat_texts = [
            g[1].lower().replace(",", " ").strip()
            for g in check_group(bnd, text, all_properties)
            if g[2] > 0
        ]
        if any(w in concat_texts for w in ["ad"]):
            return True
        lum_iou_areas, lum_other_areas = [], []
        for g_i, g_text, g_iou, g_category, g_prop_bnd, g_fg_lum in check_group(
            bnd, text, all_properties
        ):
            if g_iou > 0:
                lum_iou_areas.append(g_fg_lum)
            elif g_prop_bnd[0][1] > 50 and g_prop_bnd[-1][1] < 2850:
                lum_other_areas.append(g_fg_lum)
        if len(lum_iou_areas) > 0 and len(lum_other_areas) > 0:
            max_lum_iou_areas = max(lum_iou_areas)
            max_lum_others = max(lum_other_areas)
            if max_lum_iou_areas - max_lum_others > 150:
                return False
        return True

    elif target == "ForcedAction-General":
        return check_pattern(patterns, "ForcedAction-General", text)

    elif target == "Obstruction-RoachMotel":
        return check_pattern(patterns, "Obstruction-RoachMotel", text)

    elif target == "II-AM-FalseHierarchy-BAD":
        concat_texts = []
        hierarchy_elements = []
        bg_lum = []
        find_groups = check_group(bnd, text, all_properties)
        for g_i, g_text, g_iou, g_category, g_prop_bnd, g_fg_lum in find_groups:
            if g_iou > 0 and g_category in ["TextView", "Button", "RadioButton"]:
                if g_text.lower().replace(",", " ").strip() == "90":
                    continue
                concat_texts.append(g_text.lower().replace(",", " ").strip())
                hierarchy_elements.append(Polygon(g_prop_bnd).area)
                bg_lum.append(g_fg_lum)
            if g_text == "ad":
                return False
        if 0 < len(concat_texts) < 5:
            concat_texts = " ".join(concat_texts)
            e1 = text_ranker.encode(text, convert_to_tensor=True)
            e2 = text_ranker.encode(concat_texts, convert_to_tensor=True)
            score = util.pytorch_cos_sim(e1, e2).tolist()[0][0]
            if score > 0.6:
                if (
                    round(max(hierarchy_elements) / min(hierarchy_elements)) >= 2
                    or max(bg_lum) - min(bg_lum) > 100
                ):
                    return True

    elif target == "Nagging":
        rect = bnb_to_width_height(bnd)
        if rect["width"] > 1100 and int(bnd[-1][1]) > 2700:
            if any(
                g_text == "x"
                for _, g_text, g_iou, _, _, _ in check_group(bnd, text, all_properties)
                if g_iou > 0
            ):
                return False
            return True
        if rect["width"] > 700 and rect["height"] > 1100:
            return True
        if any(
            g_text == "X"
            for _, g_text, g_iou, _, _, _ in check_group(bnd, text, all_properties)
            if g_iou > 0
        ):
            return True

    elif target == "II-AM-Tricked":
        if text == "[this element does not contain texts]".lower() or check_pattern(
            patterns, "II-AM-Tricked", text
        ):
            return True

    elif target == "ForcedAction-SocialPyramid":
        return check_pattern(patterns, "ForcedAction-SocialPyramid", text)

    elif target == "ForcedAction-Gamification":
        return check_pattern(patterns, "ForcedAction-Gamification", text)

    elif target == "II-AM-General":
        bbnd_full_img = [[1, 1], [1, 2960], [1440, 2906], [1440, 1]]
        iou = calculate_iou(bnd, bbnd_full_img, "other")
        if iou < 0.005 and text in [
            "x",
            "skip",
            "0",
            "[this element does not contain texts]".lower(),
        ]:
            return True

    return ret


def load_img_properties(img_ids, dataset_path):
    img_all_properties = {}
    f_img_all_properties = glob(f"{dataset_path}/*_all_properties.json")

    for img_id in img_ids:
        _id = img_id[:-4]
        file_path = f"{dataset_path}/{_id}_all_properties.json"
        if file_path in f_img_all_properties:
            with open(file_path, "r") as f:
                img_all_properties[img_id] = json.load(f)
        else:
            img_all_properties[img_id] = []
    return img_all_properties


def determine_refinement(
    y_probs_high, text, total_labels, bnd, last_check_bait_and_switch, img_id
):
    y_tmp = np.zeros(len(y_probs_high))

    if (
        sum(y_probs_high) == 1
        and y_probs_high[total_labels.index("II-Preselection")] != 1
        and y_probs_high[total_labels.index("ForcedAction-Privacy")] != 1
    ):
        return y_probs_high

    if all(
        y_probs_high[total_labels.index(label)] == 1
        for label in ["ForcedAction-Privacy", "II-HiddenInformation", "II-Preselection"]
    ):
        return y_probs_high

    if all(
        y_probs_high[total_labels.index(label)] == 1
        for label in ["Sneaking-ForcedContinuity", "II-HiddenInformation"]
    ):
        return y_probs_high

    if (
        text == "[this element does not contain texts]"
        and y_probs_high[total_labels.index("II-Preselection")] == 1
    ):
        y_tmp[total_labels.index("II-Preselection")] = 1
        return y_tmp

    for i, y_idx in enumerate(y_probs_high):
        if y_idx > 0:
            ybar = total_labels[i]
            if rule_check(text, img_all_properties.get(img_id, []), ybar, bnd):
                y_tmp[i] = 1

            if ybar == "Sneaking-BaitAndSwitch":
                next_img_id = find_next_img(img_id)
                if next_img_id:
                    last_check_bait_and_switch.append([pred_idx, img_id, next_img_id])

    adjust_labels_based_on_rules(y_tmp, total_labels)
    return y_tmp


def adjust_labels_based_on_rules(y_tmp, total_labels):
    index_map = {label: total_labels.index(label) for label in total_labels}

    if y_tmp[index_map["Sneaking-ForcedContinuity"]] == 1:
        y_tmp[index_map["II-AM-FalseHierarchy-BAD"]] = 0
        y_tmp[index_map["II-Preselection"]] = 0
        y_tmp[index_map["II-HiddenInformation"]] = 1

    if y_tmp[index_map["II-AM-FalseHierarchy-BAD"]] == 1:
        y_tmp[index_map["Nagging"]] = 0
        y_tmp[index_map["ForcedAction-Privacy"]] = 0

    if y_tmp[index_map["Obstruction-RoachMotel"]] == 1:
        y_tmp[index_map["II-Preselection"]] = 0
        y_tmp[index_map["Nagging"]] = 0

    if y_tmp[index_map["II-AM-DisguisedAD"]] == 1:
        y_tmp[index_map["II-Preselection"]] = 0
        y_tmp[index_map["II-AM-FalseHierarchy-BAD"]] = 0

    if y_tmp[index_map["Obstruction-Currency"]] == 1:
        y_tmp[index_map["II-Preselection"]] = 0
        y_tmp[index_map["II-AM-FalseHierarchy-BAD"]] = 0

    if y_tmp[index_map["II-AM-ToyingWithEmotion"]] == 1:
        y_tmp[index_map["II-Preselection"]] = 0
        y_tmp[index_map["II-AM-FalseHierarchy-BAD"]] = 0

    if y_tmp[index_map["ForcedAction-General"]] == 1:
        y_tmp[index_map["II-Preselection"]] = 0
        y_tmp[index_map["II-HiddenInformation"]] = 0
        y_tmp[index_map["II-AM-DisguisedAD"]] = 0
        y_tmp[index_map["II-AM-ToyingWithEmotion"]] = 0
        y_tmp[index_map["Sneaking-BaitAndSwitch"]] = 0

    if y_tmp[index_map["ForcedAction-SocialPyramid"]] == 1:
        y_tmp[index_map["II-Preselection"]] = 0
        y_tmp[index_map["II-HiddenInformation"]] = 0

    if y_tmp[index_map["Nagging"]] == 1:
        y_tmp[index_map["Sneaking-ForcedContinuity"]] = 0
        y_tmp[index_map["ForcedAction-General"]] = 0
        y_tmp[index_map["II-Preselection"]] = 0

    if y_tmp[index_map["Sneaking-BaitAndSwitch"]] == 1:
        y_tmp[index_map["Nagging"]] = 0

    if (
        sum(
            y_tmp[index_map[label]]
            for label in [
                "ForcedAction-Privacy",
                "II-HiddenInformation",
                "Sneaking-ForcedContinuity",
            ]
        )
        == 3
    ):
        y_tmp[index_map["II-Preselection"]] = 1

    if y_tmp[index_map["II-AM-General"]] == 1:
        y_tmp[index_map["Sneaking-BaitAndSwitch"]] = 0


def apply_bait_and_switch_refinement(
    last_check_bait_and_switch, total_labels, y_refines, test_dataset
):
    for pred_idx, baitswitch_img_id, next_img_id in last_check_bait_and_switch:
        next_img_lb = find_img_label(
            next_img_id, train_dataset, test_dataset, y_refines, "label"
        )
        if "Nagging" in next_img_lb:
            y_refines[pred_idx][total_labels.index("Sneaking-BaitAndSwitch")] = 1


def print_metrics(y_trues, y_refines, total_labels):
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_trues, y_refines, average="micro", zero_division=np.nan
    )
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_trues, y_refines, average="macro", zero_division=np.nan
    )

    print(p_micro, r_micro, f1_micro, p_mac, r_mac, f1_mac)

    num_classes = y_trues.shape[1]

    for i in range(num_classes):
        class_name = total_labels[i]
        precision = precision_score(y_trues[:, i], y_refines[:, i])
        recall = recall_score(y_trues[:, i], y_refines[:, i])
        f1 = f1_score(y_trues[:, i], y_refines[:, i])
        correct_predictions = np.sum(
            (y_trues[:, i] == y_refines[:, i]) & (y_trues[:, i] == 1), axis=0
        )

        print(
            class_name,
            precision,
            recall,
            f1,
            correct_predictions,
            np.sum(y_trues[:, i]),
        )


def refine_predictions(
    pred_results,
    best_threshold,
    low_threshold,
    total_labels,
    dataset_path,
    test_dataset,
):
    img_ids = [test_dataset.data_set[p[0]][0] for p in pred_results]
    img_all_properties = load_img_properties(img_ids, dataset_path)

    y_refines, y_trues, others_props = [], [], []
    last_check_bait_and_switch = []
    high_threshold = best_threshold

    for pred_idx, result in enumerate(pred_results):
        idx, y_probs, y_true = result[:3]
        y_probs_low = np.where(np.array(y_probs) > low_threshold, y_probs, 0).tolist()
        y_probs_high = [1 if p >= high_threshold else 0 for p in y_probs]
        y_trues.append(y_true)

        y_true_text = [total_labels[i] for i, l in enumerate(y_true) if l == 1]
        (
            img_id,
            sub_img,
            bnd,
            f_full_img,
            text,
            all_text,
            one_hot_label,
        ) = test_dataset.data_set[idx]
        bnd = [list(map(int, sublist)) for sublist in bnd]
        all_properties = img_all_properties.get(img_id, [])

        if text == "[this element does not contain texts]":
            find_groups = find_check_group(bnd, text, all_properties)
            cont_text = [
                g[1].lower().replace(",", " ").strip() for g in find_groups if g[2] > 0
            ]
            text = " ".join(cont_text) if cont_text else text

        others_props.append([idx, img_id, bnd, text, y_probs_low, y_probs_high])

        y_refines.append(
            determine_refinement(
                y_probs_high,
                text,
                total_labels,
                bnd,
                last_check_bait_and_switch,
                img_id,
            )
        )

    apply_bait_and_switch_refinement(
        last_check_bait_and_switch, total_labels, y_refines, test_dataset
    )

    y_trues, y_refines = np.array(y_trues), np.array(y_refines)
    print_metrics(y_trues, y_refines, total_labels)


@click.command()
@click.option(
    "--input-pred", default="pred_floats.pk", type=str, help="Input a predict file."
)
def main(input_pred):
    _, our_test_set, total_labels, _ = load_dataset(
        OUR_DATASET_IMG_TEXT, OUR_DATASET_PROC_DATA, OUR_DATASET_IMGS
    )
    total_labels = list(set(total_labels))
    tatal_labels = sorted(total_labels)

    our_test_set = one_hot_vector(our_test_set, tatal_labels)
    test_dataset = CustomDataset(all_test_set, tokenizer, max_seq_length)

    with open(input_pred, "rb") as f:
        pred_results = pk.load(f)
    _, best_threshold, _, _, _, _ = calculate_scores(pred_results)
    low_threshold = best_threshold * 0.95

    refine_predictions(
        pred_results,
        best_threshold,
        low_threshold,
        total_labels,
        dataset_path,
        test_dataset,
    )


if __name__ == "__main__":
    main()
