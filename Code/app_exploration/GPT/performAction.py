import os
import re
import json
import ast
import time
import xml.etree.ElementTree as ET


def get_action(response):
    print("Response:", response)
    content = response.choices[0].message.content
    # content = content.replace("'", "\"")
    try:
        return ast.literal_eval(content)
    except Exception as e:
        print("Failed to parse action")
        print(content)
        raise e
        
def capture_screenshot_viewhierarchy(folder, index, device):
    time_view = time.time()
    png_filename = os.path.join(folder, f"{index}.png")
    device.screenshot(png_filename)
    print("Capturing screenshot:", png_filename)

    xml_filename = os.path.join(folder, f"{index}.xml")
    xml = device.dump_hierarchy()
    with open(xml_filename, "w") as f:
        f.write(xml)
    print("Capturing view hierarchy:", xml_filename)
    time_view = time.time() - time_view
    return os.path.abspath(xml_filename), time_view


def perform_action(action, tap_id_position_map, DEVICE_SERIAL, device):
    print("Performing action")
    success = True
    semantic_action = None
    action_points = []
    
    if action["action"] == "tap":
        _, action_points = click_element_by_id(action["id"], tap_id_position_map, DEVICE_SERIAL)
        semantic_action = "CLICK"
        action_points = [action_points]
    elif action["action"] == "type":
        id2text = action["id2text"]
        id2text = list(id2text.items())
        id2text = sorted(id2text,key=lambda x:int(x[0]))
        # for action_id, text in id2text:#[::-1]:
        #     _, tmp_points = click_element_by_id(action_id, tap_id_position_map, DEVICE_SERIAL)
        #     input_text(text, DEVICE_SERIAL)
        #     # enter_action(DEVICE_SERIAL) # doing this will incur the error message and make the other input action with a wrong coordinate
        #     action_points.append([*tmp_points, text, "KEYCODE_ENTER"])
        performType(action_points, id2text,  tap_id_position_map, DEVICE_SERIAL, device)
        enter_action(DEVICE_SERIAL)
        semantic_action = "CLICK_AND_TYPE"
         
    elif action["action"] == "scroll":
        _, action_points = scroll(action["scroll-reference"], action["direction"], tap_id_position_map, DEVICE_SERIAL)
        if action["direction"] == "up":
            semantic_action = "SCROLL_DOWN_UP"
        elif action["direction"] == "down":
            semantic_action = "SCROLL_UP_DOWN"
        elif action["direction"] == "left":
            semantic_action = "SCROLL_RIGHT_LEFT"
        elif action["direction"] == "right":
            semantic_action = "SCROLL_RIGHT_LEFT"
    elif action["action"] == "back":
        back_action(DEVICE_SERIAL)
        semantic_action = 'BACK'
    elif action["action"] == "enter":
        enter_action(DEVICE_SERIAL)
    elif action["action"] == "wait":
        print("Wait for UI Rendering")
    elif action["action"] == "turnOffSwitch":
        for idx in action["ids"]:
            _, tmp_points = click_element_by_id(idx, tap_id_position_map, DEVICE_SERIAL)
            action_points.append(tmp_points)
        semantic_action = "CLICK"
    else:
        success = False
    return success, semantic_action, action_points


def input_text(text, DEVICE_SERIAL):
    cmd_clear = f"adb -s {DEVICE_SERIAL} shell input keyevent KEYCODE_MOVE_END"
    os.system(cmd_clear)
    print(cmd_clear)
    cmd_clear = "adb -s " + DEVICE_SERIAL + " shell input keyevent --longpress $(printf 'KEYCODE_DEL %.0s' {1..50})"
    os.system(cmd_clear)
    print(cmd_clear)

    text = text.replace(" ", "%s")
    cmd_entertext = f"""adb -s {DEVICE_SERIAL} shell input text \"{text}\""""
    os.system(cmd_entertext)
    print(cmd_entertext)

def scroll(scroll_id, direction, scroll_id_position_map, DEVICE_SERIAL):
    action_points = []
    if direction not in {"up", "down", "left", "right"}:
        print(f"Invalid scroll direction: {direction}")
        return False, action_points
    
    if scroll_id not in scroll_id_position_map:
        print("Invalid Scrollable ID")
        return False, action_points

    pos = scroll_id_position_map[scroll_id]
    x = pos["x"]
    y = pos["y"]

    dx = {
        "up": 0,
        "down": 0,
        "left": 300,
        "right": -300,
    }

    dy = {
        "up": 500,
        "down": -500,
        "left": 0,
        "right": 0,
    }
    cmd_scroll = f"adb -s {DEVICE_SERIAL} shell input swipe {x} {y} {x+dx[direction]} {y+dy[direction]} 100"
    os.system(cmd_scroll)
    print(cmd_scroll)
    return True, [[x,y], [x+dx[direction], y+dy[direction]]]


def back_action(DEVICE_SERIAL):
    os.system(f"adb -s {DEVICE_SERIAL} shell input keyevent 4")


def enter_action(DEVICE_SERIAL):
    os.system(f"""adb -s {DEVICE_SERIAL} shell input keyevent 66""")


def click_element_by_id(id, tap_id_position_map, DEVICE_SERIAL):
    if id in tap_id_position_map:
        pos = tap_id_position_map[id]
        ty = pos["type"]
        if ty == "other":
            print("Tempting to click a non-actionable element")
        return click_location(pos["x"], pos["y"], DEVICE_SERIAL), [pos["x"], pos["y"]]
    return False, []


def click_location(x, y, DEVICE_SERIAL):
    cmd_tap = f"adb -s {DEVICE_SERIAL} shell input tap {x} {y}"
    os.system(cmd_tap)
    print(cmd_tap)


def performType(action_points, id2text,  tap_id_position_map, DEVICE_SERIAL, device):
    for index, (action_id, act_text) in enumerate(id2text):
        if index == 0:
            _, tmp_points = click_element_by_id(action_id, tap_id_position_map, DEVICE_SERIAL)
            input_text(act_text, DEVICE_SERIAL)
            action_points.append([*tmp_points, act_text, "KEYCODE_ENTER"])
        else:
            curr_elem = tap_id_position_map[action_id]

            # find new location
            text = curr_elem["text"]
            resource_id = curr_elem["resource-id"]
            content_desc = curr_elem["cont-desc"]
            

            xml = device.dump_hierarchy()
            tree = ET.fromstring(xml)

            target_coordinate = []

            for elem in tree.iter():
                bounds = elem.attrib.get("bounds")
                if bounds and len(elem) == 0:
                    matches = re.findall(r"-?\d+", bounds)[:4]
                    matches = [int(a) for a in matches]
                    x1,y1,x2,y2 = matches
                    points = [(x1+x2)//2, (y1+y2)//2]

                    if resource_id and len(resource_id)>0:
                        if elem.get("resource-id") == resource_id:
                            target_coordinate = points
                            break
                    
                    if content_desc and len(content_desc)>0:
                        if elem.get("content-desc") == content_desc:
                            target_coordinate = points
                            break

                    if text and len(text) > 0:
                        if elem.get("text") == text:
                            target_coordinate = points
                            break
            if len(target_coordinate)>0:            
                click_location(target_coordinate[0],target_coordinate[1], DEVICE_SERIAL)
                input_text(act_text, DEVICE_SERIAL)
                action_points.append([*target_coordinate, act_text, "KEYCODE_ENTER"])


