import json
import time
import os
import subprocess
import uiautomator2 as u2
from datetime import datetime
import traceback

# from tasks import app2tasks, apkx2cmd
from elementParsing import get_view_hierarchy
from Request2GPT import ask_gpt
from performAction import get_action, perform_action

from Config import GPT_OUTPUT_FOLDER, DEVICE_SERIAL
## setup 
import sys
from app_tasklist import app2tasks

OUTPUT_FOLDER = GPT_OUTPUT_FOLDER

ui_title_template = """
<table class=\"table\">
<tr><th>package</th><td>{}</td></tr>
<tr><th>activity</th><td>{}</td></tr>
<tr><th>state_str</th><td>{}</td></tr>
<tr><th>structure_str</th><td>{}</td></tr>
<tr><th>task</th><td>{}</td></tr>
"""

MAX_ACTION_NUM = 15
def capture_screenshot_viewhierarchy(folder, index, device):
    time_view = time.time()
    png_filename = os.path.join(folder, f"{index}.png")
    device.screenshot(png_filename)
    print("Capturing screenshot:", png_filename)


    xml_filename = os.path.join(folder, f"{index}.xml")
    # subprocess.run("adb shell uiautomator dump", shell=True)
    # subprocess.run(f'adb pull /sdcard/window_dump.xml "{xml_filename}"', shell=True)
    xml = device.dump_hierarchy()
    with open(xml_filename, "w") as f:
        f.write(xml)
    print("Capturing view hierarchy:", xml_filename)
    time_view = time.time() - time_view

    return os.path.abspath(xml_filename), time_view

def save_utg_to_file(utg_pkg, output_utg_filename):
    with open(output_utg_filename, "w") as f:
        f.write("var utg = \n")
        f.write(json.dumps(utg_pkg, indent=2))

def addNode(nodes, device, folder, index, hierarchy_filename, task):
    currActiv = device.app_current()["activity"]
    print("Current activity:", currActiv)
    nodes.append({
        "id": f'{os.path.basename(folder)}_{index}',
        "shape": "image",
        "image": hierarchy_filename.replace(".xml", ".png"),
        "title": ui_title_template.format(package, currActiv, index, index, task),
        "label": currActiv,
        "package": package,
        "activity": currActiv,
        "state_str": f'{os.path.basename(folder)}_{index}',
        "structure_str": f'{os.path.basename(folder)}_{index}',
        "step_ids": [],
    })


def restart_app(device, package):
    os.system(f"adb -s {DEVICE_SERIAL} shell pm disable {package} ")
    time.sleep(1)
    os.system(f"adb -s {DEVICE_SERIAL} shell pm enable {package} ")
    time.sleep(1)
    device.app_start(package, use_monkey=True) 
    # time.sleep(1)
    time.sleep(10)


def addEdge(edges, folder, index, hierarchy_filename, event_type, action_points, inputText=""):
    edges.append({
        "from": f'{os.path.basename(folder)}_{index-1}',
        "to": f'{os.path.basename(folder)}_{index}',
        "id": f'{os.path.basename(folder)}_{index-1}-->{os.path.basename(folder)}_{index}',
        "events": [{
            "event_str": f'{os.path.basename(folder)}_{index-1}',
            "event_id": 1,
            "event_type": event_type,
            "view_images": [hierarchy_filename.replace(".xml", ".png"), hierarchy_filename.replace(f"{index-1}.xml", f"{index}.png")],
            "action_points": action_points,
            "inputText": inputText,
        },]
    })  
            
def perform_task(task, folder, device, package, appName, maxSteps):
    index = 0
    history = []
    previous_xml = None
    nodes = []
    edges = []

    while index < maxSteps:
        time.sleep(8)
        # png_filename = os.path.join(folder, f"{index}.png")
        hierarchy_filename, time_view = capture_screenshot_viewhierarchy(folder, index, device)
        stripped_view, tap_id_position_map, time_process_vh = get_view_hierarchy(hierarchy_filename)#, previous_xml, history)

        time_askGPT = time.time()
        response = ask_gpt(appName, history, stripped_view, task, previous_xml)
        # print(response["choices"][0]["message"])

        action = get_action(response)
        time_askGPT = time.time() - time_askGPT

        action["activity"] = device.app_current()
        action["action_points"] = []
        action["time_view"] = time_view
        action["time_process_vh"] = time_process_vh
        action["time_askGPT"] = time_askGPT


        addNode(nodes, device, folder, index, hierarchy_filename, task)
        
        index += 1
        if action["action"] == "stop" or index >= maxSteps:
            if action["action"] == "stop":
                history.append(action)
                save_actions(os.path.join(folder, "actions.json"), task, history)
            else:
                # perform the last action
                time_performAction = time.time()
                _, semantic_action, action_points = perform_action(action, tap_id_position_map, DEVICE_SERIAL, device)
                time.sleep(2)
                action["action_points"] = action_points
                if semantic_action == "CLICK_AND_TYPE":
                    capture_screenshot_viewhierarchy(folder, f"{index-1}.1", device)
                    os.system(f"""adb -s {DEVICE_SERIAL} shell input keyevent 66""")
                # capture the final page
                hierarchy_filename, time_view = capture_screenshot_viewhierarchy(folder, index, device)

                print("NEXT ACTION: ", semantic_action)
                print("ACTION PARAMETER: ", action_points)
                addEdge(edges, folder, index, hierarchy_filename, event_type=semantic_action, action_points=[action_points])
                time_performAction = time.time() - time_performAction
                action["time_performAction"] = time_performAction

                history.append(action)
                save_actions(os.path.join(folder, "actions.json"), task, history)
            break
        
        
        time_performAction = time.time()
        _, semantic_action, action_points = perform_action(action, tap_id_position_map, DEVICE_SERIAL, device)

        if semantic_action == "CLICK_AND_TYPE":
            capture_screenshot_viewhierarchy(folder, f"{index-1}.1", device)
            os.system(f"""adb -s {DEVICE_SERIAL} shell input keyevent 66""")

        print("NEXT ACTION: ", semantic_action)
        print("ACTION PARAMETER: ", action_points)
        addEdge(edges, folder, index, hierarchy_filename, event_type=semantic_action, action_points=[action_points])
        
        action["action_points"] = action_points
        history.append(action)
        time_performAction = time.time() - time_performAction
        action["time_performAction"] = time_performAction
        save_actions(os.path.join(folder, "actions.json"), task, history)

        previous_xml = stripped_view

    time.sleep(8)
    return nodes, edges


def launch_app(package, device):
    device.app_start(package, use_monkey=True) 

def save_actions(filename, task, actions):
    with open(filename, "w") as f:
        out = {"goal": task, "actions": actions}
        f.write(json.dumps(out, indent=2))


def check_current_activity(package, device):
    # check if we still on the app page
    cmd_activity = f"adb -s {DEVICE_SERIAL} shell dumpsys activity top"
    activity_output = subprocess.run(cmd_activity, shell=True, capture_output=True, text=True)
    curr_activity = str(activity_output).split("mResumed=true")[0].split("\\n")[-3].strip().split(" ")[1].split("/")[0]
    if curr_activity != package.lower():
        launch_app(package.lower(), device)
    time.sleep(2)


def run_test(package, device, utg_pkg):
    output = os.path.join(OUTPUT_FOLDER, package)

    while os.path.exists(output):
        output += "1"
    os.makedirs(output)
    print("CREATING OUTPUT FOLDER:", output)

    output_utg_filename = output + ".js"

    task_names = app2tasks[package]["tasks"]
    appName = app2tasks[package]["appName"]
    for task in task_names:
        print(task)
        restart_app(device, package)

        case_output = os.path.join(output, task.replace(" ", "_"))
        if not os.path.exists(case_output):
            os.makedirs(case_output)
        print("==> CREATE TASK FOLDER:", case_output)

        try:
            max_steps = MAX_ACTION_NUM
            if "Go shopping, select any product you l" in task:
                max_steps = 20
            nodes, edges = perform_task(task, case_output, device, package, appName, max_steps)
            utg_pkg["nodes"].extend(nodes)
            utg_pkg["edges"].extend(edges)
            save_utg_to_file(utg_pkg, output_utg_filename)
        except Exception:
            error = traceback.format_exc()
            print(error)
            with open(os.path.join(case_output, "error.log"), "w") as f:
                f.write(error)
            print("Task failed")
        else:
            print("Task completed")
    return output_utg_filename


def restart():
    print("Restarting device")
    subprocess.run(f"adb -s {DEVICE_SERIAL} -e reboot", shell=True)
    time.sleep(300)

def timediffrence(from_date, to_date):
    time_format = "%Y-%m-%d %H:%M:%S.%f"
    time1 = datetime.strptime(from_date, time_format)
    time2 = datetime.strptime(to_date, time_format)
    time_spent = (time2 - time1).total_seconds() * 1000
    return time_spent

if __name__ == "__main__": 
    # connect to Android device
    device = u2.connect(DEVICE_SERIAL)
    print("## START TESTING")
    PACKAGES = list(app2tasks.keys())
    PACKAGES.sort()
    for package in PACKAGES:
        # os.system(f"adb -s {DEVICE_SERIAL} shell pm disable {package} ")
        # continue
        print("+++", package, "+++")
        # output_utg_filename = package + ".js"
        utg_pkg = {"nodes": [],
                   "edges": [],
                    "device_serial": "",
                    "device_sdk_version": "30",
                    "app_sha256": "",
                    "device_model_number": "Genymobile_Pixel 3 XL_11",
                    "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                   }
        
        # try:
        output_utg_filename = run_test(package, device, utg_pkg)
        utg_pkg["end_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        utg_pkg["time_spent"] = timediffrence(utg_pkg["test_date"], utg_pkg["end_date"])
        utg_pkg["num_edges"] = len(utg_pkg["edges"])
        utg_pkg["num_nodes"] = len(utg_pkg["nodes"])
        
        save_utg_to_file(utg_pkg, output_utg_filename)

            
        os.system(f"adb -s {DEVICE_SERIAL} shell pm disable {package} ")
        time.sleep(1)
            

