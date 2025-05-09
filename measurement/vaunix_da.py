import os 
from ctypes import *

os.add_dll_directory('C:\\_Lib\\python\\meas\\digital_att_files')
vnx = cdll.VNX_atten64

def get_vaunix_atten(channel):
# Open the dll

    # Set test mode to false
    # This means that we will be using real devices
    vnx.fnLDA_SetTestMode(False)

    # Get the number of devices
    devices_num = vnx.fnLDA_GetNumDevices()
    print('Number of devices: ', devices_num)
    # Create an array of device ids for connected devices
    DeviceIDArray = c_int * devices_num
    devices_list = DeviceIDArray()
    # fill the array with the ID's of connected attenuators
    vnx.fnLDA_GetDevInfo(devices_list)

    if len(devices_list) > 0:
        # Select which device to use
        devid = 0
        if len(devices_list) == 1:
            devid = devices_list[0]
        else:
            while not devid in devices_list:
                print("Connected Devices:")
                for device in devices_list:
                    print(f"\t({device}) {vnx.fnLDA_GetSerialNumber(device)}")
                try:
                    devid = int(input("Select a device: "))
                    if not devid in devices_list:
                        print("Invalid device selection")
                except ValueError:
                    print("Invalid device selection")
                print()
    else:
        raise RuntimeError("No devices found")
        
    # Open selected device
    vnx.fnLDA_InitDevice(devid)
    # Get some basic information about device that we will need for later
    minAttenuation = int(vnx.fnLDA_GetMinAttenuation(devid) / 4)
    maxAttenuation = int(vnx.fnLDA_GetMaxAttenuation(devid) / 4)
    numChannels = vnx.fnLDA_GetNumChannels(devid)
    print('Device ID: ', devid)
    print('Serial Number: ', vnx.fnLDA_GetSerialNumber(devid))
    print('Min Attenuation: ', minAttenuation)
    print('Max Attenuation: ', maxAttenuation)
    print('Number of Channels: ', numChannels)

    return devid

def set_atten(devid, channel, attenuation):
    vnx.fnLDA_SetChannel(devid, channel)
    vnx.fnLDA_SetAttenuationHR(devid, int(attenuation * 20))
    print('Current device set to channel ', str(channel), ' attenuation ', str(attenuation))

def get_atten(devid):
    data = vnx.fnLDA_GetAttenuationHR(devid)
    return (data/20.0)