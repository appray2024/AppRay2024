## clean xml and get hash value for deduplication
import os
import re
import json
import ast
from glob import glob
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter
import random
from PIL import Image, ImageFile

# This line will allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True


def hash_string(input_string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_string.encode())
    return sha256_hash.hexdigest()


def find_leaf_nodes(element, leaf_list):
    if len(element) == 0:  # If the current node has no child nodes, it's a leaf
        leaf_list.append(element)
    else:
        for child in element:
            find_leaf_nodes(child, leaf_list)

def get_leaf_nodes(root, target_pkg):
    leaf_nodes = []
    find_leaf_nodes(root, leaf_nodes)
    # print(len(leaf_nodes))

    leaf_node_info = []
    for node in leaf_nodes:
        res = node.attrib.get("resource-id", "")
        tag = node.attrib.get("class", "")
        pkg = node.attrib.get("package", "") # package is been deleted in previous step
        # print("pacakge:", pkg, target_pkg)
        # print()
        # if pkg != target_pkg:
        #     continue

        leaf_node_info.append(ET.tostring(node, encoding="utf-8").decode("utf-8").strip())
    return leaf_node_info

# Function to remove attributes from an element
def remove_attributes(element, attributes):
    # fastbot2 does not capture "visible-to-user" attrib
    visible = element.attrib.get("visible-to-user", "true")
    if visible != "true":
        # del element
        return 1
    for attr in attributes:
        if attr in element.attrib:
            del element.attrib[attr]
    res = element.attrib.get("resource-id")
    if res == "android:id/statusBarBackground":# or "com.android.systemui" in str(res):
        # remove system status bar. It may appear multiple times
        # print("statusBar", element.attrib.get("bounds"), xml_file_path)
        # del element
        return 1
    
    ## R2: remove view without text
    class1 = element.attrib.get("class", "")
    text = element.attrib.get("text", "")
    if len(element)==0 and class1 == "android.view.View" and len(text) == 0:
        return 1

    bounds = element.attrib.get("bounds")
    # print("bounds", bounds)
    if bounds is None:
        # del element
        return 1
    x,y,x2,y2 = [int(x) for x in re.findall(r'-?\d+', bounds)]
    w, h = x2-x, y2-y
    # R1
    if w<=0 or h<=0 or x<0 or y<0 or x2<0 or y2<0 or y>2960 or y2-y <=5:
        element.attrib["flag_remove"] = True
    # if not(0<=x<1440 and 0<=y<2960 and 0<x2<=1440 and 0<y2<=2960):
    #     element.attrib["flag_remove"] = True
    del element.attrib["bounds"]

    # R3: remove layout node all attributes
    if "Layout" in element.attrib["class"] or "ViewGroup" in element.attrib["class"] or "android.widget.ScrollView" in element.attrib["class"]:
        del element.attrib["class"]
        del element.attrib["resource-id"]

# Function to recursively remove attributes from child elements
def remove_attributes_recursively(element, attributes_to_remove):
    delete_id = []
    for child_idx, child in enumerate(element):
        flag = remove_attributes(child, attributes_to_remove)
        if flag:
            delete_id.append(child_idx)
            continue
        remove_attributes_recursively(child, attributes_to_remove)

    for child_idx in delete_id[::-1]:
        del element[child_idx]

def remove_invisible_elements(element):
    removed_index = []
    for idx, child in enumerate(element):
        if child.attrib.get("flag_remove", False):
            removed_index.append(idx)
            continue
        remove_invisible_elements(child)
    for idx in removed_index[::-1]:
        del element[idx]

def get_all_packageName(currentPackageName, root):
    for idx, child in enumerate(root):
        package = child.attrib.get("package", "")
        if package != "":
            currentPackageName.append(package)


def merge_child(Iparent, child_id=-1):
    if child_id == -1:
        curr_node = Iparent
    else:
        curr_node = Iparent[child_id]
    
    if len(curr_node) == 1:
        while len(curr_node) == 1:
            curr_node = curr_node[0]
        Iparent[child_id] = curr_node
    if len(curr_node) == 0:
        return
    else:
        # >1
        del_child_id = []
        for new_child_id in range(len(curr_node)):
            new_child = curr_node[new_child_id]
            if len(new_child) == 0 and new_child.attrib.get("bounds") == "[0,0][1440,2960]":
                del_child_id.append(new_child_id)
                continue
            merge_child(curr_node, new_child_id)
        for new_child_id in del_child_id[::-1]:
            del curr_node[new_child_id]

def getXMLhash(xml_file_path):
    currentPackageName = [] 
    if "fastbot" in xml_file_path:
        target_pkg = xml_file_path.split("/")[-2].split("-")[1]
    else:
        target_pkg = xml_file_path.split("/")[-3].strip("1")
    with open(xml_file_path, "r") as xml_file:
        xml_content = xml_file.read()
    # Parse the XML content
    root = ET.fromstring(xml_content)
    # print(root, len(root))
    random.seed(123)

    # delete system notification (pop-up notification and the dropdown notification page that are not visible)
    del_id = []
    for idx in range(len(root)):
        # remove the framelayout for system notification
        tmp_node = root[idx]
        currPack = tmp_node.attrib["package"]
        if currPack == "com.android.systemui":
            del_id.append(idx)
    for idx in del_id[::-1]:
        del root[idx]    

    # some xml only capture notification
    if len(root) == 0:
        # print(xml_file_path)
        return 1, 1

    # List of attributes to remove
    attributes_to_remove = ["index", "package", "checkable", "enabled",  "focusable", "focused", "scrollable", "long-clickable", "password", "selected", "visible-to-user", "checked", "clickable", "content-desc", "text", "NAF"]#, , "bounds"]#,  

    # Remove attributes from the root element
    remove_attributes(root, attributes_to_remove)

    # Recursively remove attributes from child elements
    remove_attributes_recursively(root, attributes_to_remove)

    remove_invisible_elements(root)

    merge_child(root)
    for parent_id, parent in enumerate(root):
        while len(parent) == 1:
            parent = parent[0]
        root[parent_id] = parent

    get_all_packageName(currentPackageName, root)

    # Counting unique items and their frequency
    counter = Counter(currentPackageName)
    xmlPackage = counter.most_common(1)[0][0] if len(counter) > 0 else ""

    # only keep the leaf node
    modified_xml_string = get_leaf_nodes(root, target_pkg)

    # Serialize the modified XML back to string
    # modified_xml_string = ET.tostring(root, encoding="utf-8").decode("utf-8")

    # if root.tag == "hierarchy":
    #     modified_xml_string="\n".join(modified_xml_string.split("\n")[1:-1])

    # Print the modified XML string
    # print(f"\n\n####\n{modified_xml_string}")

    hashed_value = hash_string("\n".join(modified_xml_string).replace(" ", ""))
    # hashed_value = "\n".join(modified_xml_string).replace(" ", "")
    return hashed_value, modified_xml_string, xmlPackage


def getPNGhash(image_path):
    # Load the image
    image = Image.open(image_path)

    # Crop the image (left, upper, right, lower)
    cropped_image = image.crop((0, 84, 1440, 2960))

    # Generate the hash for the cropped  image
    image_byte_array = cropped_image.tobytes()
    hash_object = hashlib.sha256(image_byte_array)  # You can use other hashing algorithms too, like sha512, md5, etc.
    hex_dig = hash_object.hexdigest()
    return [hex_dig]

if __name__=="__main__":
    hash1 = getPNGhash("xxx.png")
    hash2 = getPNGhash("yyy.png")
    print(hash1 == hash2)

    xmls = ["xxx.xml", 
            "yyy.xml"
        ]

    xmlHash1, str1 = getXMLhash(xmls[0])[:2]
    xmlHash2, str2 = getXMLhash(xmls[1])[:2]
    print(xmlHash1==xmlHash2)

    str1 = set(str1)
    str2 = set(str2)

    same = str1.intersection(str2)
    diff1 = str1 - str2
    diff2 = str2 - str1
    print("SAME:\n", "\n".join(list(same)))
    print("DIFF1:\n", "\n".join(list(diff1)))
    print("DIFF2:\n", "\n".join(list(diff2)))

    tok1 = str1.replace(" ", "").split("\n")
    tok2 = str2.replace(" ", "").split("\n")
    print("\n\n")
    for i in range(max(len(tok1), len(tok2))):
        print(f"{i}: {tok1[i]==tok2[i]}")
        print("1", tok1[i])
        print("2", tok2[i])