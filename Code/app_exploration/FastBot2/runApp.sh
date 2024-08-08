#!/bin/bash


serial="127.0.0.1:6555" #44  
outputFolder="ui_exploration_fastbot_output"
package_list=(
                # "com.google.android.apps.translate"
                "com.shazam.android"
                "com.ubercab"
                "com.guardian"
                "com.woolworths"
)


for new_package in "${package_list[@]}"; do
    cmd1="adb -s $serial shell monkey -p $new_package -c android.intent.category.LAUNCHER 1"
    eval "$cmd1"

    time sleep 8

    # 构建新的命令并执行
    cmd="adb -s $serial shell \
    CLASSPATH=/sdcard/monkeyq.jar:/sdcard/framework.jar:/sdcard/fastbot-thirdpart.jar exec app_process /system/bin com.android.commands.monkey.Monkey \
    -p $new_package \
    --agent reuseq \
    --running-minutes 10 \
    --throttle 1500 -v -v > $outputFolder/$new_package.log"

    echo "执行命令："
    echo "$cmd"
    echo ""

    # 执行命令
    eval "$cmd"

    echo "命令执行完毕。"
    echo ""
done


