''' Extract the interaction trace, UI info from Fastbot log
Input: a log file
Output: js file (without deduuplication)
'''

import re
import os
from glob import glob
import json
import hashlib
import xml.etree.ElementTree as ET

from datetime import datetime

ui_title_template = """
<table class=\"table\">
<tr><th>package</th><td>{}</td></tr>
<tr><th>activity</th><td>{}r</td></tr>
<tr><th>state_str</th><td>{}</td></tr>
<tr><th>structure_str</th><td>{}</td></tr>
"""
      
def remove_continuous_duplicates(lst):
    if not lst:
        return []

    result = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            result.append(lst[i])
    return result
def getActionPoints(rawActionPoints):
    action_points = []
    for line in rawActionPoints:
        match = re.match(r".*:Sending (Touch|Key) \(ACTION_(DOWN|UP|MOVE)\): (\d+)\s*(.*)?", line)
        if match:
            action_type, action_code, _, action_description = match.groups()
            # print(action_type, action_code, action_description)
            if "KEYCODE_" in action_description:
                # KEY
                # Sending Key (ACTION_DOWN): 4    // KEYCODE_BACK
                # action_description: // KEYCODE_BACK
                action_description = [action_description[3:]]
            else:
                # Touch
                # Sending Touch (ACTION_DOWN): 0:(720.0,1480.0)
                pattern = r"\(\d+.\d+,\d+.\d+\)"
                point_matches = re.findall(pattern, action_description)
                action_points_tmp = []
                for point in point_matches:
                    point = point.strip()[1:-1].split(",")
                    x_value = int(float(point[0]))
                    y_value = int(float(point[1]))
                    currPoints = [x_value, y_value]
                    action_points_tmp.append(currPoints)
                action_description = action_points_tmp

            if action_code == "DOWN":
                # first move for each action
                # :Sending Touch (ACTION_DOWN): 0:(720.0,1480.0)
                action_points.append(action_description)
            else:
                # not-first move
                action_points[-1].extend(action_description)

    # remove duplicate action points in each action
    for idx in range(len(action_points)):
        action_points[idx] = remove_continuous_duplicates(action_points[idx])

    return action_points

def getUTG(input_log_filename):
    output_jsname = input_log_filename.replace(".log", ".js")
    print("OUTPUT FILEName:", output_jsname)

    outputFolder = os.path.abspath(os.path.dirname(output_jsname))

    # Read the content from the file
    with open(input_log_filename, 'r') as file:
        content = file.readlines()

    utg = {
        "nodes": [],
        "edges": [],
        "num_nodes": 0,
        "num_edges": 0,
        "num_transitions": 0,
        "device_serial": "",
        "device_sdk_version": "30",
        "app_sha256": "",
    }
    indexTotal, indexVisit, indexEnd = None, None, None
    currPNG, currAction, currActiv, currUID, currInputText = [None] * 5
    currIntent = None
    currActionPoints = []
    prevPNG, prevAction, prevActiv, prevUID = [None] * 4
    currInputText = ""
    for idx, line in enumerate(content):
        line = line.strip()
        # print(line)

        if "@Version" in line:
            utg["test_date"] = line.split("]")[1][1:]
        elif "AllowPackage" in line:
            utg["app_package"] = line.split("AllowPackage:")[-1].strip()
        elif "Switch" in line and "component=" in line:
            utg["app_main_activity"] = line.split("component=")[1].split("/")[1].split(";")[0]
        elif "phone info" in line:
            utg["device_model_number"] = line.split("phone info：")[1].strip()
        elif "current activity" in line:  
            currActiv = line.split(" ")[-1]
            utg["num_transitions"] += 1
        elif "action type" in line:
            currAction = line.split(" ")[-1]
            # print(prevActiv, currActiv, currAction)
        elif "Saving screen shot" in line:
            currPNG = line.split("/sdcard/")[1].split(' ')[0]
            currPNG = os.path.join(outputFolder, currPNG)
            currUID = currPNG.split("-")[-1].replace(".png", "")

            if currActiv is None:
                currActiv = prevActiv

            # save current node
            node = {
                    "id": currUID,
                    "shape": "image",
                    "image": currPNG,
                    "title": ui_title_template.format(utg["app_package"], currActiv, currUID, currUID), 
                    "label": currActiv.split('.')[-1],
                    "package": utg["app_package"],
                    "activity": currActiv,
                    "state_str": currUID,
                    "structure_str": currUID,
                    "step_ids": [currPNG]
                }
            utg["nodes"].append(node)

            if prevAction is not None:
                actionPoints = getActionPoints(currActionPoints)

                edge_id = f"{prevUID}-->{currUID}"
                edge = {
                    "from": prevUID,
                    "to": currUID,
                    "id": edge_id,
                    "events": [
                        {
                            "event_str": currIntent.split(".")[-1] if currIntent is not None else currIntent,
                            "event_id": 1,
                            "event_type": prevAction,
                            "view_images": [prevPNG, currPNG], 
                            "action_points": actionPoints,
                            "inputText": currInputText,
                        }
                    ]
                }
                utg["edges"].append(edge)
                currInputText = ""
                currActionPoints = []
                currIntent = None
            prevPNG, prevAction, prevActiv, prevUID = currPNG, currAction, currActiv, currUID, 
            currPNG, currAction, currUID, currActiv = [None] * 4
        elif "Sending Touch" in line or "Sending Key" in line:
            currActionPoints.append(line)
        elif "Allowing start of Intent" in line:
            currIntent = line.split("Allowing start of")[1]
        elif "Input text is " in line:
            currInputText = line.split("Input text is")[1].strip()
        elif "Total app activities" in line:
            indexTotal = idx
        elif "Explored app activities" in line: 
            indexVisit = idx
        elif "Activity of Coverage" in line:
            indexEnd = idx
    if currActiv is not None:
        node = {
                "id": currUID,
                "shape": "image",
                "image": currPNG,
                "title": ui_title_template.format(utg["app_package"], currActiv, currUID, currUID), 
                "label": currActiv.split('.')[-1],
                "package": utg["app_package"],
                "activity": currActiv,
                "state_str": currUID,
                "structure_str": currUID,
                "step_ids": [currPNG]
            }
        utg["nodes"].append(node)
            
    all_activities = content[indexTotal: indexVisit]
    all_activities = list(map(lambda x:x.strip().split(" ")[-1], all_activities))
    if indexEnd:
        explored_activities = content[indexVisit:indexEnd]
        explored_activities = list(map(lambda x:x.strip().split(" ")[-1], explored_activities))
        utg["num_reached_activities"] = len(explored_activities)
        utg["num_effective_events"] = utg["num_reached_activities"]

        utg["end_date"] = content[indexEnd].strip().split("]")[1][1:]

    utg["num_nodes"] = len(utg["nodes"])
    utg["num_edges"] = len(utg["edges"])

    utg["app_num_total_activities"] = len(all_activities)

    time_format = "%Y-%m-%d %H:%M:%S.%f"
    time1 = datetime.strptime(utg["test_date"], time_format)
    if indexEnd:
        time2 = datetime.strptime(utg["end_date"], time_format)
        utg["time_spent"] = (time2 - time1).total_seconds() * 1000


    # Convert the utg dictionary to a JSON format
    with open(output_jsname, "w") as f:
        f.write("var utg = \n")
        f.write(json.dumps(utg, indent=2))


if __name__ == "__main__":
    inputFolder = "ui_exploration_fastbot_output"

    all_logs = glob(os.path.join(inputFolder, "**.log"))
    for input_log_filename in all_logs:
        getUTG(input_log_filename)