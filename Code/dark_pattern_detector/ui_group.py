import json
import numpy as np
from PIL import Image, ImageDraw
from glob import glob
import os
import cv2
import shutil
import click


IMAGE_WIDTH = 1440
IMAGE_HEIGHT = 2960

# Function to parse JSON and return UI elements
def parse_json(json_file):
    with open(json_file, "r") as file:
        try:
            return json.load(file)
        except json.decoder.JSONDecodeError as e:
            print(f"Error reading {json_file}: {e}")
            raise


# Function to group elements horizontally
def group_elements_horizontally(ui_elements, vertical_offset=50):
    sorted_elements = sorted(ui_elements, key=lambda x: x["bbox"][1])
    groups = []
    current_group = []
    max_height = 0

    for elem in sorted_elements:
        _, ymin, _, ymax = elem["bbox"]
        if not current_group or ymin <= max_height + vertical_offset:
            current_group.append(elem)
            max_height = max(max_height, ymax)
        else:
            groups.append(current_group)
            current_group = [elem]
            max_height = ymax

    if current_group:
        groups.append(current_group)

    return groups


# Function to group elements by containment
def group_elements_by_containment(ui_elements):
    groups = []
    for large_elem in ui_elements:
        contained_elements = [large_elem]
        lxmin, lymin, lxmax, lymax = large_elem["bbox"]

        for small_elem in ui_elements:
            if small_elem == large_elem:
                continue
            sxmin, symin, sxmax, symax = small_elem["bbox"]
            if sxmin >= lxmin and sxmax <= lxmax and symin >= lymin and symax <= lymax:
                contained_elements.append(small_elem)

        if len(contained_elements) > 1:
            groups.append(contained_elements)

    return groups


# Function to group elements vertically
def group_elements_vertically(ui_elements, vertical_threshold=50):
    sorted_elements = sorted(ui_elements, key=lambda x: x["bbox"][1])
    groups = []
    current_group = []
    last_ymax = 0

    for elem in sorted_elements:
        _, ymin, _, ymax = elem["bbox"]
        if not current_group or ymin <= last_ymax + vertical_threshold:
            current_group.append(elem)
            last_ymax = ymax
        else:
            groups.append(current_group)
            current_group = [elem]
            last_ymax = ymax

    if current_group:
        groups.append(current_group)

    return groups


def calculate_group_bbox(elements):
    bboxes = np.array([e["bbox"] for e in elements])
    xmin, ymin = np.min(bboxes[:, 0]), np.min(bboxes[:, 1])
    xmax, ymax = np.max(bboxes[:, 2]), np.max(bboxes[:, 3])
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def resize_final_bbox_width(bbox):
    xmin, ymin, xmax, ymax = bbox
    if (xmax - xmin) > (0.66 * IMAGE_WIDTH):
        return [xmin, ymin, IMAGE_WIDTH - 10, ymax]
    return bbox


# Function to check contrast
def check_contrast(image, bbox, contrast_threshold=30):
    x1, y1, x2, y2 = bbox
    element_crop = image[y1:y2, x1:x2]
    background_color = np.mean(image[:50, :50], axis=(0, 1))
    element_color = np.mean(element_crop, axis=(0, 1))
    contrast = np.linalg.norm(background_color - element_color)
    return contrast < contrast_threshold


def brightness_region(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    roi = image[y_min:y_max, x_min:x_max]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


# Function to check if two bounding boxes overlap fully
def do_boxes_overlap(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    return (
        x1_min <= x2_max and x1_max >= x2_min and y1_min <= y2_max and y1_max >= y2_min
    )


# Function to merge overlapping groups
def merge_overlapping_groups(groups):
    merged_groups = []
    merged_texts = []

    while groups:
        group = groups.pop(0)
        group_texts = [
            txt_elem["text"]
            for elem in group
            if "text_items" in elem
            for txt_elem in elem["text_items"]
        ]
        bbox1 = calculate_group_bbox(group)
        merged = False

        for i, other_group in enumerate(merged_groups):
            bbox2 = calculate_group_bbox(other_group)
            if do_boxes_overlap(bbox1, bbox2):
                merged_groups[i].extend(group)
                merged_texts[i].extend(group_texts)
                merged = True
                break

        if not merged:
            merged_groups.append(group)
            merged_texts.append(group_texts)

    return merged_groups, merged_texts


def calculate_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    return max(0, x_max - x_min) * max(0, y_max - y_min)


def calculate_overlap_area(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    overlap_x_min = max(x1_min, x2_min)
    overlap_y_min = max(y1_min, y2_min)
    overlap_x_max = min(x1_max, x2_max)
    overlap_y_max = min(y1_max, y2_max)

    if overlap_x_min < overlap_x_max and overlap_y_min < overlap_y_max:
        return calculate_area(
            (overlap_x_min, overlap_y_min, overlap_x_max, overlap_y_max)
        )
    return 0


def check_percent_overlap(bbox1, bbox2, threshold=0.8):
    area1 = calculate_area(bbox1)
    area2 = calculate_area(bbox2)
    overlap_area = calculate_overlap_area(bbox1, bbox2)
    smaller_area = min(area1, area2)
    larger_area = max(area1, area2)
    return (
        overlap_area >= threshold * smaller_area,
        larger_area / smaller_area,
        bbox1 if area1 > area2 else bbox2,
    )


def bbox_to_points(bnd):
    return [[bnd[0], bnd[1]], [bnd[2], bnd[1]], [bnd[2], bnd[3]], [bnd[0], bnd[3]]]


def point_convert_rect(points):
    return [int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[-1][1])]


def bnb_to_width_height(bnd):
    return {
        "width": int(bnd[1][0]) - int(bnd[0][0]),
        "height": int(bnd[-1][1]) - int(bnd[0][1]),
    }


def merge_gth_dp_file(gth_dp_file, image, ret):
    img_DPs = parse_json(gth_dp_file)
    for dp in img_DPs:
        points, text, labels = dp
        bnd = point_convert_rect(points)
        for prop in ret["properties"]:
            is_overlap, ratio, larger_bnd = check_percent_overlap(bnd, prop["bbox"])
            if is_overlap:
                prop["labels"].extend(labels)
                prop["text"].append(text)
                prop["bbox"] = larger_bnd if ratio < 2 or larger_bnd == bnd else bnd
        cv2.rectangle(image, (bnd[0], bnd[1]), (bnd[2], bnd[3]), (0, 0, 255), 20)
        cv2.putText(
            image,
            " ".join(labels),
            (bnd[0] + 10, bnd[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            2,
        )
    return ret


# Function to draw bounding boxes on image
def generate_group_json(
    image_path,
    groups,
    merged_groups_texts,
    gth_dp_file,
    output_img_path,
    output_properties_path,
    color=(255, 0, 0),
    thickness=10,
):
    image_name = os.path.basename(image_path)
    app_name = image_name.split("_")[0]
    ret = {
        "ori_img_path": image_path,
        "image": image_name,
        "app_name": app_name,
        "properties": [],
    }

    image = cv2.imread(image_path)
    for i, bbox in enumerate(groups):
        set_merged_groups_texts = list(set(merged_groups_texts[i]))
        prop = {"bbox": bbox, "text": set_merged_groups_texts, "labels": []}
        ret["properties"].append(prop)

    if os.path.exists(gth_dp_file):
        ret = merge_gth_dp_file(gth_dp_file, image, ret)

    page_bboxes = []
    num_clear = 0
    _props = []

    for prop in ret["properties"]:
        bbox_str = " ".join(map(str, prop["bbox"]))
        if bbox_str in page_bboxes:
            idx = page_bboxes.index(bbox_str)
            _prop = _props[idx]
            _prop["text"].extend(prop["text"])
            _prop["labels"].extend(prop["labels"])
            continue

        page_bboxes.append(bbox_str)
        prop["bbox"] = bbox_to_points(prop["bbox"])
        prop["text"] = list(set(prop["text"]))

        bnd = point_convert_rect(prop["bbox"])
        brightness = brightness_region(image, bnd)
        if brightness < 150 and not prop["labels"]:
            continue

        if not prop["labels"]:
            num_clear += 1
            prop["labels"] = ["CLEAR"]

        cv2.rectangle(image, (bnd[0], bnd[1]), (bnd[2], bnd[3]), color, thickness)
        cv2.putText(
            image,
            " ".join(prop["labels"]).strip(),
            (bnd[0], bnd[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            color,
            2,
        )
        _props.append(prop)

    ret["properties"] = _props
    cv2.imwrite(output_img_path, image)
    with open(output_properties_path, "w") as f:
        json.dump(ret, f)

    return num_clear


@click.command()
@click.option(
    "--image-path", default="/data/scsgpu1/work/jeffwang/dark_pattern/our_labelling_v2/round_4/ui_group/user_study", type=str, help="Root path of images."
)
def main(image_path):
    root_path = image_path
    image_root = os.path.join(root_path, "imgToBeExamined")
    frcnn_json_root = os.path.join(root_path, "all_properties_paddle")
    ui_group_root = os.path.join(root_path, "ui_group")
    ui_group_images = os.path.join(ui_group_root, "group_images")
    ui_group_properties = os.path.join(ui_group_root, "group_properties")
    gth_dp_properties_root = os.path.join(frcnn_json_root, "text")

    os.makedirs(ui_group_images, exist_ok=True)
    os.makedirs(ui_group_properties, exist_ok=True)
    os.makedirs(gth_dp_properties_root, exist_ok=True)

    if "user_study" in root_path:
        image_names = [
            os.path.basename(img_name)
            for img_name in glob(os.path.join(image_root, "*.png"))
        ]
        f_dp_labels = [
            os.path.join(
                root_path,
                "all_properties_paddle",
                img.replace(".png", "") + "_dp_checker.json",
            )
            for img in image_names
        ]

        for i, f_dp in enumerate(f_dp_labels):
            img = image_names[i]
            f_ph_subimg_prop = os.path.join(
                gth_dp_properties_root, img + "_subimg_text_ph.json"
            )

            if not os.path.exists(f_ph_subimg_prop):
                img_ph_subimg_prop = []
                with open(f_dp, "r") as f:
                    dp = json.load(f)
                    for label, values in dp.items():
                        for v in values:
                            bbox = bbox_to_points(v["bbox"])
                            text = " ".join(
                                child["text"]
                                for child in v.get("children", [])
                                if "text" in child
                            ).strip()
                            img_ph_subimg_prop.append([bbox, text, [label]])
                with open(f_ph_subimg_prop, "w") as f:
                    json.dump(img_ph_subimg_prop, f)

    image_files = glob(os.path.join(image_root, "*.png"))
    image_names = [os.path.basename(f) for f in image_files]
    frcnn_json_files = [
        os.path.join(frcnn_json_root, img.replace(".png", "_all_properties.json"))
        for img in image_names
    ]
    gth_dp_files = [
        os.path.join(gth_dp_properties_root, img + "_subimg_text_ph.json")
        for img in image_names
    ]
    all_text_files = [
        os.path.join(gth_dp_properties_root, img + "_all_text_ph.json")
        for img in image_names
    ]

    output_group_img_files = [
        os.path.join(ui_group_images, f"group_{img}") for img in image_names
    ]
    output_group_properties_files = [
        os.path.join(ui_group_properties, img.replace(".png", "") + "_group.json")
        for img in image_names
    ]

    total_num_clears = 0
    for i in range(len(image_files)):
        json_file = frcnn_json_files[i]
        image_file = image_files[i]
        output_image_path = output_group_img_files[i]
        output_properties_path = output_group_properties_files[i]

        ui_elements = parse_json(json_file)

        if os.path.exists(all_text_files[i]):
            dst = os.path.join(ui_group_properties, os.path.basename(all_text_files[i]))
            shutil.copyfile(all_text_files[i], dst)
        else:
            all_texts = [
                [elem["text"], elem["bbox"]] for elem in ui_elements if "text" in elem
            ]
            with open(all_text_files[i], "w") as f:
                json.dump(all_texts, f)

        horizontal_groups = group_elements_horizontally(ui_elements)
        vertical_groups = group_elements_vertically(ui_elements)
        containment_groups = group_elements_by_containment(ui_elements)

        image_cv = cv2.imread(image_file)
        all_groups = horizontal_groups + vertical_groups + containment_groups

        merged_groups, merged_groups_texts = merge_overlapping_groups(all_groups)
        all_bounding_boxes = [
            resize_final_bbox_width(calculate_group_bbox(group))
            for group in merged_groups
            if resize_final_bbox_width(calculate_group_bbox(group)) is not None
        ]

        num_clears = generate_group_json(
            image_file,
            all_bounding_boxes,
            merged_groups_texts,
            gth_dp_files[i],
            output_image_path,
            output_properties_path,
        )
        total_num_clears += num_clears

    print(f"Total number of clears: {total_num_clears}")


if __name__ == "__main__":
    main()
