

# Usage
Please refer to https://github.com/bytedance/Fastbot_Android for configuration setup

## add cmake to path so that we can access via terminal
PATH="xxx/Library/Android/sdk/cmake/3.22.1/bin":"$PATH"
PATH="xxx/Library/Android/sdk/ndk/25.2.9519653":"$PATH"
PATH="xxx/Library/Android/sdk/cmake/3.22.1/bin/ninja":"$PATH"
NDK_ROOT=xxx/Library/Android/sdk/ndk/25.2.9519653

need to install cmake, ninja, ndk
set the cmake.dir and sdk.dir in local.properties file

## add a line to build_native.sh
NDK_ROOT=xxx/Library/Android/sdk/ndk/25.2.9519653

install ADBKeyboard and set it as default keyboard 
https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk

adb push monkeyq.jar /sdcard/monkeyq.jar
adb push fastbot-thirdpart.jar /sdcard/fastbot-thirdpart.jar
adb push libs/* /data/local/tmp/
adb push framework.jar /sdcard/framework.jar
adb push max.config /sdcard


## collect UIs using fastbot2
bash runApp.sh

## saved collected UIs from device
bash pullFromDevice.sh