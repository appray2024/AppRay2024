''' Deduplication for GPT4.
The input file is the action file for each task and the output is  a js file
'''
## merge steps in gpt
import os
import json
from glob import glob
from checkXMLhash import getXMLhash
from utils_finer import UI_TITLE_Template, getTempID_Both



def main(all_actions, output_jsname):
    # iterate the actions.json files
    app_folder_packageName = all_actions[0].split("/")[-3].strip("1")
    all_actions.sort()
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

    # go through all uis in one app. Only keep unique nodes and edges, other duplicates info will be saved to node["duplicateIDs"], and edge["events"]
    xmlHash2PNGPath = {}
    all_edgeIDs2index = {}
    last_node = None
    all_nodeIDs2node = {}
    for action_file in all_actions:
        if "5-close" in action_file.lower():
            # skip this task. no need. should be captured by other tasks
            continue
        with open(action_file, "r") as f:
            actions = json.load(f)
        
        task_folder = os.path.dirname(action_file)
        pad = 0
        for ui_id, action in enumerate(actions["actions"]):
            # print(f"current Task {os.path.basename(task_folder)[0:2]}:", ui_id) # debug
            if isinstance(action, list):
                action = action[0]
            duplicate2path = None
            ui_path = os.path.join(task_folder, f"{ui_id}.png")
            xml_path = os.path.join(task_folder, f"{ui_id}.xml")


            if not os.path.exists(ui_path):
                pad += 1
            ui_id += pad
            # print("ui_id:", ui_id)
            ui_path = os.path.join(task_folder, f"{ui_id}.png")
            xml_path = os.path.join(task_folder, f"{ui_id}.xml")

            # if current ui is duplicate, just add the link to that ui
            currUID = ui_path
            currUID_dup = ui_path
                                                                                                                                   
            # check if the current ui is already been visited
            hashOutput = getXMLhash(xml_path)
            currXMLHash = hashOutput[0]

            # if getTempID_Both(currUID) in ["T3-U15", "T4-U2"]: # for debug
            #     print("##", getTempID_Both(currUID))
            #     print(currXMLHash)

            # print("getTempID_Both(currUID)", getTempID_Both(ui_path), currXMLHash)
            if currXMLHash in xmlHash2PNGPath:
                duplicate2path = xmlHash2PNGPath[currXMLHash]
                currUID = duplicate2path
                # print("rm node:", getTempID_Both(ui_path), getTempID_Both(duplicate2path))
            else:
                xmlHash2PNGPath[currXMLHash] = ui_path

            package = action.get("activity", {"package": None})["package"]
            activity = action.get("activity", {"activity": None})["activity"]
            

            if duplicate2path is None:
                node = {
                        "id": getTempID_Both(currUID),
                        "shape": "image",
                        "image": ui_path,
                        "title": UI_TITLE_Template.format(package, activity, getTempID_Both(currUID), currUID.replace(".png", ".xml")), 
                        "label": activity.split('.')[-1],
                        "package": package,
                        "activity": package+activity,
                        "state_str": getTempID_Both(currUID),
                        "duplicateIDs": [],
                        # "structure_str": currUID.replace(".png", ".xml"),
                        # "step_ids": [ui_path],
                        # "flag_duplicate": 0 if duplicate2path is None else 1,
                        # "duplicate2file": duplicate2path,
                    }
                all_nodeIDs2node[getTempID_Both(currUID)] = node
                utg["nodes"].append(node)
            else:
                # print(currUID, currUID_dup)
                existing_node = all_nodeIDs2node[getTempID_Both(currUID)]
                existing_node["duplicateIDs"].append(getTempID_Both(currUID_dup))


            action_type = action["action"]
            if action_type == "scroll":
                action_type+="_"+action["direction"]

            # add an edge from current node to next node
            actionPoints = action.get("action_points")
            next_ui_path = os.path.join(task_folder, f"{ui_id+1}.png")
            next_xml_path = os.path.join(task_folder, f"{ui_id+1}.xml")
            
            if os.path.exists(next_ui_path):
                nextUID = next_ui_path
                nextUID_dup = nextUID

                # check if next ui already exist
                nextXMLHash = getXMLhash(next_xml_path)[0]
                if nextXMLHash == 1:
                    break
                if nextXMLHash in xmlHash2PNGPath:
                    nextUID = xmlHash2PNGPath[nextXMLHash]

                # if getTempID_Both(nextUID) in ["T3-U15", "T4-U2"]: # for debug
                #     print("next_ui_path:", next_ui_path) #debug
                #     print("##", getTempID_Both(nextUID))
                #     print(nextXMLHash)
                #     print("\n\n")
                
                # check if the edge is duplicate
                # if it is, just skip it
                edge_id = f"{getTempID_Both(currUID)}-->{getTempID_Both(nextUID)}"
                if edge_id in all_edgeIDs2index:
                    # add to event
                    edge = utg["edges"][all_edgeIDs2index[edge_id]]
                    edge["events"].append({
                        "event_str": f'{getTempID_Both(currUID_dup)}->{getTempID_Both(nextUID_dup)}', # no use
                        "event_id": len(edge["events"])+1,
                        "event_type": action_type,
                        "view_images": [currUID_dup, nextUID_dup], 
                        "action_points": actionPoints,
                        "inputText": None, # no use
                    })
                    continue
                all_edgeIDs2index[edge_id] = len(utg["edges"])
                edge = {
                    "from": getTempID_Both(currUID),
                    "to": getTempID_Both(nextUID),
                    "id": edge_id,
                    "events": [
                        {
                            "event_str": f'{getTempID_Both(currUID_dup)}->{getTempID_Both(nextUID_dup)}',
                            "event_id": 1,
                            "event_type": action_type,
                            "view_images": [currUID_dup, nextUID_dup], 
                            "action_points": actionPoints,
                            "inputText": None, # no use
                        }
                    ]
                }
                utg["edges"].append(edge)
                last_node = {
                        "id": getTempID_Both(nextUID),
                        "shape": "image",
                        "image": nextUID,
                        "title": UI_TITLE_Template.format("", "", getTempID_Both(nextUID), nextUID.replace(".png", ".xml")), 
                        "label": "",
                        "package": "",
                        "activity": "",
                        "state_str": getTempID_Both(nextUID),
                        "duplicateIDs": [getTempID_Both(nextUID_dup)] if nextUID_dup!=nextUID else [],
                    }
                # if getTempID_Both(nextUID_dup) in ["T3-U15"]: # for debug
                #     tmpXMLHash = getXMLhash(nextUID_dup.replace(".png", ".xml"))[0]
                #     print("##", getTempID_Both(nextUID_dup))
                #     print(tmpXMLHash)
                # print(getTempID_Both(nextUID_dup), getTempID_Both(nextUID)) #debug
        all_nodeids = list(map(lambda x:x["id"], utg["nodes"]))
        if last_node is not None:
            if last_node["id"] not in all_nodeids:
                utg["nodes"].append(last_node)
                last_node_xml = last_node["image"].replace(".png", ".xml")
                all_nodeIDs2node[last_node["id"]] = last_node
                xmlHash2PNGPath[getXMLhash(last_node_xml)[0]] = last_node["image"]
            else:
                # print(last_node["id"], last_node) # debug
                # print(all_nodeids, "\n\n")# debug
                if len(last_node["duplicateIDs"]) >0:
                    last_duplicateIDs = last_node["duplicateIDs"][0]
                    uniqueNode = [node for node in utg["nodes"] if node["id"] == last_node["id"]][0]
                    if last_duplicateIDs not in uniqueNode["duplicateIDs"]:
                        uniqueNode["duplicateIDs"].append(last_duplicateIDs)

    
    utg["num_edges"] = len(utg["edges"])
    utg["num_nodes"] = len(utg["nodes"])

    # Convert the utg dictionary to a JSON format
    with open(output_jsname, "w") as f:
        f.write("var utg = \n")
        f.write(json.dumps(utg, indent=2))



def sort_task_by_their_running_order(js_files):
    # Desired order of folder numbers
    desired_order = [9, 10, 3, 4, 5, 7, 8]

    # Create a mapping from folder number to its position in the desired order
    order_mapping = {num: index for index, num in enumerate(desired_order)}

    # Extract the numeric part from folder names and sort according to the desired order
    def custom_sort_key(folder):
        # Extract the numeric part from the folder name
        folder_number = int(os.path.basename(os.path.dirname(folder)).split('-')[0])
        # Return the position of the folder number in the desired order
        return order_mapping.get(folder_number, float('inf'))  # Use infinity for numbers not in desired order

    # Sort the folders using the custom sort key
    sorted_folders = sorted(js_files, key=custom_sort_key)
    return sorted_folders


if __name__=="__main__":
    
    folder = "ui_exploration_gpt_output"
    output_folder = "ui_exploration_gpt_output-deduplication"

    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_package_folder = glob(folder + "/**")
    all_package_folder.sort()
    for pkg_folder in all_package_folder:
        if not os.path.isdir(pkg_folder):
            continue
        package = os.path.basename(pkg_folder).strip("1")

        output_jsname = os.path.join(output_folder, package+".js")
        all_actions = glob(pkg_folder+"/**/**.json")

        main(all_actions, output_jsname)

        print(output_jsname)
        