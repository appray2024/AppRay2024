
import xml.etree.ElementTree as ET
import re
import ast
import subprocess
import os
import time
from Request2GPT import ask_gpt


def download_view_hierarchy(filename):
    print("Downloading view hierarchy")
    subprocess.run("adb shell uiautomator dump", shell=True)
    subprocess.run(f'adb pull /sdcard/window_dump.xml "{filename}"', shell=True)

    # adb shell uiautomator dump /sdcard/window_dump.xml & adb pull /sdcard/window_dump.xml mainScreen.xml

def check_dupicate_or_invalid_actions(root, previous_xml, history):
    
    stripped = ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8").replace("\n", "").replace("\r", "")
    if len(history) == 0:
        return root
    if stripped == previous_xml:
        # {'action': 'scroll', 'scroll-reference': '0', 'direction': 'down', 'reason': 'Scrolling down to find the settings page'}
        scroll_id = history[-1].get("scroll-reference",-1)
        tap_id = history[-1].get("id",-1)
        for elem in root.iter():
            # TODO: when it comes to scrolling, we need to only remove one of the directions and retain others
            if elem.attrib.get("scroll-reference", -2) == scroll_id:
                elem.attrib.pop("scroll-reference")
            if elem.attrib.get("id", -2) == tap_id:
                elem.attrib.pop("id")
            if elem.attrib.get("other-id", -2) == tap_id:
                elem.attrib.pop("other-id")
    return root



# def get_view_hierarchy(filename, previous_xml, history):
def get_view_hierarchy(filename):
    time_process_vh = time.time()
    tree = ET.parse(filename)
    root = tree.getroot()

    remove_attribs = [
        "index",
        "package",
        "checkable",
        # "checked",
        "focusable",
        # "focused",
        "password",
        # "selected",
        "enabled",
        "scrollable",
        "resource-id",
        "NAF", # Not Accessibility Friendly
        "bounds",
        "clickable",
        "rotation",
        "long-clickable",
        "class",
        "content-desc",
        "visible-to-user",
    ]

    # Remove unnecessary elements
    def attribute_or(a,b):
        if "true" in [a,b]:
            return "true"
        return "false"

    for idx in range(len(root)):
        # remove the framelayout for system notification
        tmp_node = root[idx]
        currPack = tmp_node.attrib["package"]
        if currPack == "com.android.systemui":
            del root[idx]
            break
    parent = root[0]


    while len(parent) == 1:
        parent[0].attrib["checkable"] = attribute_or(parent[0].attrib.get("checkable", "false"), parent.attrib.get("checkable", "false"))
        parent[0].attrib["clickable"] = attribute_or(parent[0].attrib.get("clickable", "false"), parent.attrib.get("clickable", "false"))         
        parent[0].attrib["scrollable"] = attribute_or(parent[0].attrib.get("scrollable", "false"), parent.attrib.get("scrollable", "false"))
        parent = parent[0]
                           
    def merge_child(Iparent, child_id=-1):
        if child_id == -1:
            curr_node = Iparent
        else:
            curr_node = Iparent[child_id]
        
        if len(curr_node) == 1:
            while len(curr_node) == 1:
                curr_node[0].attrib["checkable"] = attribute_or(curr_node[0].attrib.get("checkable", "false"), curr_node.attrib.get("checkable", "false"))
                curr_node[0].attrib["clickable"] = attribute_or(curr_node[0].attrib.get("clickable", "false"), curr_node.attrib.get("clickable", "false"))         
                curr_node[0].attrib["scrollable"] = attribute_or(curr_node[0].attrib.get("scrollable", "false"), curr_node.attrib.get("scrollable", "false"))
                curr_node = curr_node[0]
            Iparent[child_id] = curr_node
        if len(curr_node) == 0:
            return
        else:
            # >1
            for new_child_id in range(len(curr_node)):
                merge_child(curr_node, new_child_id)

    merge_child(parent)
    root[0] = parent


    # with open(filename.replace(".xml", ".stripped-tmp.xml"), "wb") as f:
    #     ET.indent(tree)
    #     tree.write(f)

    tap_index = 0
    tap_id_position_map = {}

    for elem in root.iter():
        bounds = elem.attrib.get("bounds")
        # if bounds:
        #     matches = re.findall(r"-?\d+", bounds)[:4]
        #     matches = [int(a) for a in matches]
        #     x1,y1,x2,y2 = matches
        #     if not (0<=x1<1440 and 0<=y1<2560 and 0<x2<=1440 and 0<y2<=2560 and x1<x2 and y1<y2):
        #         root.remove(elem)
        #         continue
        ele_info = ""
        class_val = elem.attrib.get("class")
        if class_val:
            # android.widget.Button -> Button
            elem.tag = re.sub("\W+", "", class_val.split(".")[-1])
            ele_info += elem.tag + ", "

        content_desc = elem.attrib.get("content-desc")
        if content_desc:
            elem.attrib["description"] = content_desc
            ele_info += content_desc + ", "

        resource_id = elem.attrib.get("resource-id")
        if resource_id:
            elem.attrib["resource"] = resource_id.split("/")[-1]
            ele_info += elem.attrib["resource"] + ","
        ele_info +=  elem.attrib.get("text", "")

        flag_id = False
        if bounds:
            # print(bounds)
            matches = re.findall(r"-?\d+", bounds)[:4]
            matches = [int(a) for a in matches]
            x1,y1,x2,y2 = matches
            # if 0<=x1<1440 and 0<=y1<2560 and 0<x2<=1440 and 0<y2<=2560 and x1<x2 and y1<y2:
            
            x = (int(matches[0]) + int(matches[2])) / 2
            y = (int(matches[1]) + int(matches[3])) / 2

            selected = elem.attrib.get("selected")
            clickable = elem.attrib.get("clickable")
            if clickable == "true" and selected!="true":
                elem.attrib["id"] = str(tap_index)
                tap_id_position_map[str(tap_index)] = {"x": x, "y": y, "ele_info": ele_info, "type": "tap"}
                tap_index += 1
                flag_id = True

            scrollable = elem.attrib.get("scrollable")
            if scrollable == "true":
                elem.attrib["scroll-reference"] = str(tap_index)
                tap_id_position_map[str(tap_index)] = {"x": x, "y": y, "ele_info": ele_info, "type": "scroll"}
                tap_index += 1
                flag_id = True

            checkable = elem.attrib.get("checkable")
            if checkable == "true":
                elem.attrib["id"] = str(tap_index)
                tap_id_position_map[str(tap_index)] = {"x": x, "y": y, "ele_info": ele_info, "type": "tap"}
                tap_index += 1
                flag_id = True

            if not flag_id:
                elem.attrib["other-id"] = str(tap_index)
                tap_id_position_map[str(tap_index)] = {"x": x, "y": y, "ele_info": ele_info, "type": "other"}
                tap_index += 1
            tap_id_position_map[str(tap_index-1)]["text"] = elem.attrib.get("text")
            tap_id_position_map[str(tap_index-1)]["resource-id"] = resource_id
            tap_id_position_map[str(tap_index-1)]["cont-desc"] = content_desc



        checkable = elem.attrib.get("checkable")
        if checkable == "false":
            elem.attrib.pop("checked", None)

        for attrib in ["focused", "selected"]:
            if elem.attrib.get(attrib) == "false":
                elem.attrib.pop(attrib)

        for key, value in elem.attrib.copy().items():
            if not value or key in remove_attribs:  # or value == "false"
                elem.attrib.pop(key)

    ### Remove invalid actions
    # check_dupicate_or_invalid_actions(root, previous_xml, history)
    
    stripped = ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8").replace("\n", "").replace("\r", "")
    # Format only the saved view, not the string representation
    with open(filename.replace(".xml", ".stripped.xml"), "wb") as f:
        ET.indent(tree)
        tree.write(f)

    time_process_vh = time.time() - time_process_vh
    return stripped, tap_id_position_map, time_process_vh


def capture_screenshot_viewhierarchy(folder, index, device):
    png_filename = os.path.join(folder, f"{index}.png")
    device.screenshot(png_filename)
    print("Capturing screenshot:", png_filename)

    xml_filename = os.path.join(folder, f"{index}.xml")
    xml = device.dump_hierarchy()
    with open(xml_filename, "w") as f:
        f.write(xml)
    print("Capturing view hierarchy:", xml_filename)
    return os.path.abspath(xml_filename)

def compare_hierarchy(vh1, vh2):
    def remove_attributes(attributes, text):
        for attr in attributes:
            pattern = rf'{attr}="[^"]*"'
            # Find all matches
            matches = re.findall(pattern, text)
            for match in matches:
                text = text.replace(match, "")
        return text

    ignoredAttributes = ["text", "content-desc"]
    vh1_new = remove_attributes(ignoredAttributes, vh1)
    vh2_new = remove_attributes(ignoredAttributes, vh2)
    if vh1_new == vh2_new:
        return True
    return False

def get_leaf_node(element, leafnode):
    # If the element has no children, it's a leaf node
    if len(element) == 0:
        # Check if the element has a 'text' attribute and print it
        if 'text' in element.attrib:
            leafnode.append(element)
    else:
        # If the element has children, recursively call the function for each child
        for child in element:
            get_leaf_node(child, leafnode)

def append_elements(elements, element2VH, view_id, stripped_view, tap_id_position_map):
    root = ET.fromstring(stripped_view)
    tmp_leafnode = []
    get_leaf_node(root, tmp_leafnode)
    for leaf in tmp_leafnode:
        if 'id' in leaf.attrib:
            ele_id = leaf.attrib["id"]
            del leaf.attrib["id"]
        elif 'other-id' in leaf.attrib:
            ele_id = leaf.attrib["other-id"]
            del leaf.attrib["other-id"]
        # elif 'scroll-reference' in leaf.attrib:
        #     ele_id = leaf.attrib["scroll-reference"]
        else:
            continue
        action_point = [tap_id_position_map[ele_id]["x"], tap_id_position_map[ele_id]["y"]]
        leaf_str = ET.tostring(leaf, encoding="utf-8", method="xml").decode("utf-8").strip()
        if leaf_str in elements:
            continue
        
        elements.append(leaf_str)
        leaf.attrib["id"] = str(len(element2VH))
        leaf_str = ET.tostring(leaf, encoding="utf-8", method="xml").decode("utf-8").strip()
        element2VH.append([leaf_str, view_id, action_point])



def scroll_save_and_ask(previousVH, folder, index, device, tap_id_position_map):
    all_vh_actions = []
    all_elements = []
    element2VH = []
    duplicate_index =[]
    # scroll_id_position_map[str(scroll_index)] = {"x": x, "y": y, "ele_info": ele_info}
    tmp_previousVH_hori = previousVH

    append_elements(all_elements, element2VH, index, previousVH, tap_id_position_map)
    scrollNum = 0
    scroll_item = None
    for direction in ["V"]: #"H"
        prev_index = index
        for idx, (_, ele_info) in enumerate(tap_id_position_map.items()):
            if ele_info["type"] != "scroll":
                continue
            # scroll horizontal
            scroll_item = ele_info
            scrollID = 0
            while True:
                x, y = ele_info["x"], ele_info["y"]

                if direction == "H":
                    # try scroll from right to left: can only do half screen
                    print("Scroll Left on ", ele_info["ele_info"])
                    os.system(f"adb shell input swipe {x} {y} {0} {y} 100")
                else:
                    print("Scroll Up on ", ele_info["ele_info"])
                    os.system(f"adb shell input swipe {x} {y} {x} {y-500} 100")
                    all_vh_actions.append([[x,y], [x,y-500]])
                    scrollNum += 1

                hierarchy_filename = capture_screenshot_viewhierarchy(folder, f"{index}.{idx}.{direction}{scrollID}", device)
                stripped_view, tap_id_position_map, time_process_vh = get_view_hierarchy(hierarchy_filename) 
                flag = compare_hierarchy(stripped_view,tmp_previousVH_hori)
                if flag:
                    duplicate_index.append(f"{index}.{idx}.{direction}{scrollID}")
                    break

                action = {
                    "from": prev_index,
                    "to": f'{index}.{idx}.{direction}{scrollID}',
                    "action_type": "SCROLL_RIGHT_LEFT",
                    "action_points": [x,y],
                }
                if direction == "V":
                    action["action_type"] = "SCROLL_DOWN_UP"
                all_vh_actions.append(action)
                tmp_previousVH_hori = stripped_view
                prev_index = f"{index}.{idx}.{direction}{scrollID}"
                append_elements(all_elements, element2VH, f'{index}.{idx}.{direction}{scrollID}', stripped_view, tap_id_position_map)
                scrollID += 1
                time.sleep(0.5)
    
    # print(duplicate_index)
    view = [ele[0] for ele in element2VH]
    print("\n".join(view))
    # response = ask_gpt([], view, "3-go to setting page, go through all notification related pages")
    # response = ast.literal_eval(response["choices"][0]["message"]["content"])
    target_element_id = 101 #int(response["id"])
    view_id = element2VH[target_element_id][1].strip()
    print("Target VIew ID:", view_id)
    for _ in range((scrollNum)*2):
        print("Scroll DOWN", (scrollNum)*2)
        x, y = scroll_item["x"], scroll_item["y"]
        os.system(f"adb shell input swipe {x} {y} {x} {y+1000} 500")

    for idx, (_, ele_info) in enumerate(tap_id_position_map.items()):
        if ele_info["type"] != "scroll":
            continue
        # scroll horizontal
        scrollID = 0
        while True:
            x, y = ele_info["x"], ele_info["y"]
            print("Scroll UP on ", ele_info["ele_info"])
            os.system(f"adb shell input swipe {x} {y} {x} {y-500} 100")

            prev_index = f"{index}.{idx}.V{scrollID}"
            print("Curr VIew ID:", prev_index)
            if view_id == prev_index:
                break
            scrollID += 1
            time.sleep(1)
        if view_id == prev_index:
            break
    action_point = element2VH[target_element_id][2]
    time.sleep(1)
    print(action_point)
    os.system(f"adb shell input tap {action_point[0]} {action_point[1]}")

if __name__ == "__main__":
    filenmae = "TaskExploration/11.xml"
    get_view_hierarchy(filenmae)