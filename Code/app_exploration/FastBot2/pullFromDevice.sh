#!/bin/bash

# 定义要替换的原始包名和替换后的列表
package_list=("com.virginaustralia.vaapp" "com.weather.forecast.channel.local")

outputFolder="ui_exploration_fastbot_output"


for new_package in "${package_list[@]}"; do
    # 构建新的命令并执行
    cmd="adb pull /sdcard/fastbot-$new_package--running-minutes-10 $outputFolder/"

    echo "执行命令："
    echo "$cmd"
    echo ""

    # 执行命令
    eval "$cmd"

    echo "命令执行完毕。"
    echo ""
done


