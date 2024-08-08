''' Deduplication for GPT4&Fastbpt.
The input file is gpt js files and fastbot js files, the output is a merged deduplicated js file. 
Added:
1. Ccompare whether the image is the same or not
2. Check whether the UI belongs to current app (flag_notAppUI)

'''
## merge fastbot and gpt
import os
import json
from glob import glob
from checkXMLhash import  getXMLhash, getPNGhash
from utils_finer import getTempID_Both, check_if_app_uis

from PIL import Image
import matplotlib.pyplot as plt


def merge(input_js_filename, output_jsname, appPackageName):
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
            
            # print(len(temp_utg["nodes"]))
    # print(len(utg["nodes"]))

    nodeID2node= {}
    for node_idx, node in enumerate(utg["nodes"]):
        nodeID2node[node["id"]] = node

    all_edges = utg["edges"]

    all_edgeIDs2index = {}
    all_nodeIDs2node = {}
    xmlHash2PNGPath = {}
    pngHash2PNGPath = {}
    new_nodes = []
    new_edges = []

    all_existing_nodeids = []
    for edgeIDX, edge in enumerate(all_edges):
        duplicate2path = None
        
        fromID = edge["from"]
        toID = edge["to"]
        fromNode = nodeID2node[fromID]
        toNode = nodeID2node[toID]

        fromIMGPath, toIMGPath =  fromNode["image"], toNode["image"]
        fromXMLPath = fromIMGPath.replace(".png", ".xml")
        toXMLPath = toIMGPath.replace(".png", ".xml")

        if not os.path.exists(fromXMLPath) or not os.path.exists(fromIMGPath):
            # print("fromXMLPath does not exist")
            continue

        ### CHECK FROM NODE, ONLY ADD FROM NODE IF IT IS NEW
        currUID = fromIMGPath
        fromXMLHash = getXMLhash(fromXMLPath)[0]
        if fromXMLHash in xmlHash2PNGPath:
            duplicate2path = xmlHash2PNGPath[fromXMLHash]
        else:
            fromPNGHash = getPNGhash(fromIMGPath)[0]
            if fromPNGHash in pngHash2PNGPath:
                duplicate2path = pngHash2PNGPath[fromPNGHash]
            else:
                pngHash2PNGPath[fromPNGHash] = fromIMGPath
    
        if duplicate2path: 
            currUID = duplicate2path
            print("duplicateID", fromNode["id"], getTempID_Both(duplicate2path))
            existing_node = all_nodeIDs2node[getTempID_Both(duplicate2path)]
            
            # need to find the existing node
            # if existing_node["id"] != fromNode["id"]:
                
            existing_node["duplicateIDs"] = existing_node.get("duplicateIDs",[])
            existing_node["duplicateIDs"].extend(fromNode.get("duplicateIDs",[]))
            existing_node["duplicateIDs"].append(fromNode["id"])
            existing_node["duplicateIDs"] = list(set(existing_node["duplicateIDs"]))

            all_existing_nodeids.append(existing_node["id"])
            all_existing_nodeids.extend(existing_node["duplicateIDs"])

        else:
            # if is a new node, just add it
            xmlHash2PNGPath[fromXMLHash] = fromIMGPath
            all_nodeIDs2node[fromID] = fromNode
            new_nodes.append(fromNode)
            all_existing_nodeids.append(fromNode["id"])
            # print("Add from", fromNode["id"])

        if not os.path.exists(toXMLPath)  or not os.path.exists(toIMGPath):
            continue
        ### CHECK EDGE
        nextUID = toIMGPath
        nextUID_dup = toIMGPath
        
        toXMLHash = getXMLhash(toXMLPath)[0]
        if toXMLHash in xmlHash2PNGPath:
            nextUID = xmlHash2PNGPath[toXMLHash]
        else:
            toPNGHash = getPNGhash(toIMGPath)[0]
            if toPNGHash in pngHash2PNGPath:
                nextUID = pngHash2PNGPath[toPNGHash]
        if nextUID != nextUID_dup:
            toNode = nodeID2node[getTempID_Both(nextUID)]
            all_existing_nodeids.append(getTempID_Both(nextUID_dup))

        edge_id = f"{getTempID_Both(currUID)}-->{getTempID_Both(nextUID)}"
        edge["from"] = getTempID_Both(currUID)
        edge["to"] = getTempID_Both(nextUID)
        edge["id"] = edge_id

        # print(edge["from"], edge["to"], edge["id"])

        if edge_id in all_edgeIDs2index:
            # if already exists, just add events
            existing_edge = new_edges[all_edgeIDs2index[edge_id]]
            existing_edge["events"].extend(edge["events"])
            # print("Existing edge")
            continue

        all_edgeIDs2index[edge_id] = len(new_edges)
        new_edges.append(edge)

    # print(all_existing_nodeids)
    for node in utg["nodes"]:
        if node["id"] not in all_existing_nodeids:
            # print(" node id ",  node["id"] )
            new_nodes.append(node)

    # delete nodes and egde
    utg["nodes"] = new_nodes
    utg["edges"] = new_edges

    utg["num_nodes"] = len(utg["nodes"])
    utg["num_edges"] = len(utg["edges"])

    print(utg["num_nodes"])


    ## check whether the node is app ui or not
    for node in utg["nodes"]:
        package = node.get("package", "")
        activity = node.get("activity", "").split(".")[-1]

        flag_skip = check_if_app_uis(package, appPackageName, activity)
        node["flag_notAppUI"] = flag_skip

    # Convert the utg dictionary to a JSON format
    with open(output_jsname, "w") as f:
        f.write("var utg = \n")
        f.write(json.dumps(utg, indent=2))


if __name__=="__main__":
    
    GPT_folder = "ui_exploration_gpt_output-deduplication"  #change name
    FastBOT_folder = "ui_exploration_fastbot_output-deduplication"  #change name
    output_folder = "ui_exploration_merged"


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_js = glob(GPT_folder+"/**.js")  
    all_js.sort()
    flag = False
    for gpt_js_file in all_js:
        # if "com.media.bestrecorder.audiorecorder" not in gpt_js_file: # debug
            # continue
        print("==>", gpt_js_file)
        fast_js_file = gpt_js_file.replace(GPT_folder, FastBOT_folder)
        appPackageName = os.path.basename(fast_js_file).replace(".js", "")
        output_jsname = os.path.join(output_folder, appPackageName+".js")
        ## order: gpt first, fastbot 
        merge([gpt_js_file, fast_js_file], output_jsname, appPackageName)


        # break