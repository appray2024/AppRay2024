''' Deduplication for FastBot. The input file is a js file, and the output is also a js file

'''## remove invalid steps in fastbot
import os
import json
from glob import glob
from checkXMLhash import getXMLhash

from utils_finer import UI_TITLE_Template, getTempID_Both, refinePath

def main(input_js_filename, output_jsname, package_folder):
    # iterate the js files
    utg = {}
    for input_file in input_js_filename:
        with open(input_file, "r") as f:
            temp_utg = "\n".join(f.readlines()[1:])
            temp_utg = json.loads(temp_utg)
            if len(utg) == 0:
                utg = temp_utg
                utg["num_edges"] = len(temp_utg["edges"])
            else:
                utg["nodes"].extend(temp_utg["nodes"])
                utg["edges"].extend(temp_utg["edges"])

                utg["num_edges"] += len(temp_utg["edges"])


    nodeID2node= {}
    for node_idx, node in enumerate(utg["nodes"]):
        nodeID2node[node["id"]] = node

    all_edges = utg["edges"]
    # sort edge by step id
    all_edges = sorted(all_edges, key=lambda x:int(os.path.basename(x["events"][0]["view_images"][0]).split("-")[1]), )
    
    all_edgeIDs2index = {}
    all_nodeIDs2node = {}
    xmlHash2PNGPath = {}
    new_nodes = []
    new_edges = []

    last_node = None
    for edgeIDX, edge in enumerate(all_edges):
        # print("+Step:", os.path.basename(edge["events"][0]["view_images"][0]).split("-")[1], os.path.basename(edge["events"][0]["view_images"][1]).split("-")[1])
        duplicate2path = None
        
        fromID = edge["from"]
        toID = edge["to"]
        fromNode = nodeID2node[fromID]
        toNode = nodeID2node[toID]

        fromIMGPath, toIMGPath =  edge["events"][0]["view_images"]
        fromIMGPath, toIMGPath = refinePath(fromIMGPath, package_folder), refinePath(toIMGPath, package_folder)
        fromXMLPath = fromIMGPath.replace(".png", ".xml")
        toXMLPath = toIMGPath.replace(".png", ".xml")

        if not os.path.exists(fromXMLPath):
            # print("fromXMLPath does not exist")
            continue

        currUID = fromIMGPath
        currUID_dup = fromIMGPath
        fromXMLHash = getXMLhash(fromXMLPath)[0]
        # print(fromXMLHash)
        if fromXMLHash in xmlHash2PNGPath:
            duplicate2path = xmlHash2PNGPath[fromXMLHash]
            currUID = duplicate2path
        else:
            xmlHash2PNGPath[fromXMLHash] = fromIMGPath

        if duplicate2path is None:
            # print("NONE")
            fromNode["id"] = getTempID_Both(currUID)
            fromNode["image"] = currUID
            fromNode["title"] = UI_TITLE_Template.format(package, fromNode["activity"], getTempID_Both(currUID), currUID.replace(".png", ".xml")), 
            fromNode["state_str"] = getTempID_Both(currUID)
            fromNode["duplicateIDs"] = []
            del fromNode["step_ids"]
            del fromNode["structure_str"]
            # fromNode["structure_str"] = fromIMGPath
            all_nodeIDs2node[getTempID_Both(currUID)] = fromNode
            new_nodes.append(fromNode)
        else:
            existing_node = all_nodeIDs2node[getTempID_Both(currUID)]
            existing_node["duplicateIDs"].append(getTempID_Both(currUID_dup))
        
        if not os.path.exists(toXMLPath):
            # print("toXMLPath does not exist")
            continue
        toXMLHash = getXMLhash(toXMLPath)[0]

        nextUID = toIMGPath
        nextUID_dup = toIMGPath
        if toXMLHash in xmlHash2PNGPath:
            nextUID = xmlHash2PNGPath[toXMLHash]
            toNode = nodeID2node[nextUID.split("-")[-1].split(".")[0]]

        edge_id = f"{getTempID_Both(currUID)}-->{getTempID_Both(nextUID)}"
        edge["from"] = getTempID_Both(currUID)
        edge["to"] = getTempID_Both(nextUID)
        # action_type = edge["events"][0]["event_type"]
        # action_points = edge["events"][0]["action_points"]
        inputText = edge["events"][0]["inputText"]
        if len(inputText) > 0:
            edge["events"][0]["event_type"] = "TYPE"


        edge["events"][0]["event_str"] = f'{getTempID_Both(currUID_dup)}->{getTempID_Both(nextUID_dup)}'
        edge["events"][0]["view_images"] = [currUID_dup, nextUID_dup]
        edge["id"] = edge_id

        toNode["id"] = getTempID_Both(nextUID)
        toNode["image"] = nextUID
        toNode["title"] = UI_TITLE_Template.format(package, toNode["activity"], getTempID_Both(nextUID), nextUID.replace(".png", ".xml")), 
        toNode["state_str"] = getTempID_Both(nextUID)

        if edge_id in all_edgeIDs2index:
            existing_edge = new_edges[all_edgeIDs2index[edge_id]]
            edge["events"][0]["event_id"] = len(existing_edge["events"]) + 1
            existing_edge["events"].append(edge["events"][0])
            # print("Existing edge")
            continue

        all_edgeIDs2index[edge_id] = len(new_edges)
        new_edges.append(edge)
        last_node = toNode


    if last_node is not None and last_node not in new_nodes:
        print(last_node["id"])
        print("Append the last node")
        fromIMGPath = last_node["image"]
        fromXMLPath = fromIMGPath.replace(".png", ".xml")
        fromXMLHash = getXMLhash(fromXMLPath)[0]
        last_node["duplicateIDs"] = []
        if fromXMLHash not in xmlHash2PNGPath:
            xmlHash2PNGPath[fromXMLHash] = fromIMGPath
            new_nodes.append(last_node)

    # delete nodes and egde
    utg["nodes"] = new_nodes
    utg["edges"] = new_edges

    utg["num_nodes"] = len(utg["nodes"])
    utg["num_edges"] = len(utg["edges"])

    # Convert the utg dictionary to a JSON format
    with open(output_jsname, "w") as f:
        f.write("var utg = \n")
        f.write(json.dumps(utg, indent=2))


if __name__=="__main__":
    
    folder = "ui_exploration_fastbot_output"
    output_folder = "ui_exploration_fastbot_output-deduplication"

    all_js = glob(os.path.join(folder, "*.js"))
    print(len(all_js))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for js in all_js:
        package = os.path.basename(js).replace(".js", "")
        package_folder = os.path.join(folder, f"fastbot-{package}--running-minutes-10")
        # if "com.medium.reader" not in js:
        #     continue
        print("++", js)
        # output_jsname = js.replace(".js", "-clean.js")
        output_jsname = os.path.join(output_folder, package+".js")
        print("output js:", output_jsname)
        main([js], output_jsname, package_folder)

