import os


UI_TITLE_Template = """
<table class=\"table\">
<tr><th>package</th><td>{}</td></tr>
<tr><th>activity</th><td>{}</td></tr>
<tr><th>state_str</th><td>{}</td></tr>
<tr><th>structure_str</th><td>{}</td></tr>
"""


def getTempID_Both(imgPath):
    ### Generate a short ID for the UI
    ## FastBot: FastBot-S{step}
    ## GPT4: T{task_id}-U{step}
    if "fastbot" in imgPath:
        step = os.path.basename(imgPath).split("-")[1]
        tmpID = f'FastBot-S{step}'
    else:
        tmpID_tokens = imgPath.split("/")[-2:]
        tmpID = f'T{tmpID_tokens[0].split("-")[0]}-U{tmpID_tokens[-1].split(".")[0]}'
    return tmpID


def refinePath(oldPath, package_folder):
    ### the path saved in the js may be wrong as I restructured the folders
    ## need to update to the right one
    return os.path.join(package_folder, os.path.basename(oldPath))


def check_if_app_uis(package, app_folder_packageName, activity):
    ### Check if the current UI belongs to the app
    flag_skip = False
    if package != app_folder_packageName:
        
        if package in ['android', 'com.android.camera2', 'com.android.dialer', 'com.android.documentsui', 'com.android.gallery3d', 'com.android.launcher3', 'com.android.messaging', 'com.android.vending', 'com.depop', 'com.ebates', 'com.media.bestrecorder.audiorecorder', 'com.twitter.android',]:
            flag_skip = True
        if package == "com.google.android.gms" and "com.google.android" not in app_folder_packageName:
            flag_skip = True
        if package == "com.android.settings" and activity in [".SubSettings", ".Settings"]:
            flag_skip = True
    return flag_skip
