from scresonators.measurement.datamanagement import SlabFile
import re
import json
import numpy as np
import yaml

"""Reopen saved data"""


def prev_data(expt_path, filename):
    temp_data_file = expt_path + "/" + filename
    with SlabFile(temp_data_file) as a:
        attrs = dict()
        for key in list(a.attrs):
            # attrs.update({key: json.loads(a.attrs[key])})
            attrs.update({key: a.attrs[key]})
        keys = list(a)
        temp_data = dict()
        for key in keys:
            temp_data.update({key: np.array(a[key])})
    return temp_data, attrs


def grab_files(pattern, file_list, pth, n=None):
    files = [file for file in file_list if re.match(pattern, file)]
    files = sorted(files)
    if n is not None:
        if len(n) == 1:
            files = files[:n]
        else:
            files = files[n[0] : n[1]]
    data_list, attr_list = [], []
    for file in files:
        data, attrs = prev_data(pth, file)
        data_list.append(data)
        attr_list.append(attrs)

    return data_list, attr_list, files


def grab_file_list(files, pth):
    files = sorted(files)
    data_list, attr_list = [], []
    for file in files:
        data, attrs = prev_data(pth, file)
        data_list.append(data)
        attr_list.append(attrs)

    return data_list, attr_list


def get_files(pattern, file_list, pth, n=None):
    files = [file for file in file_list if re.match(pattern, file)]
    files = sorted(files)
    if n is not None:
        if len(n) == 1:
            files = files[:n]
        else:
            files = files[n[0] : n[1]]
    return files


def save_np(params, file_name):
    params_list = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            params_list[key] = value.tolist()
        else:
            params_list[key] = value

    with open(file_name, "w") as modified_file:
        yaml.dump(params_list, modified_file, default_flow_style=None)


def load(file_name):
    with open(file_name, "r") as file:
        cfg = yaml.safe_load(file)  # turn it into an attribute dictionary

    for key, value in cfg.items():
        if isinstance(value, list):
            cfg[key] = np.array(value)
    return cfg


def get_params(yml_file, meas, pth=""):
    import yaml
    import os

    with open(yml_file, "r") as file:
        sample_dict = yaml.safe_load(file)

    matching_keys = [
        key
        for key, items in sample_dict.items()
        for item in items["meas"]
        if item == meas
    ]
    sample = matching_keys[0] if matching_keys else None

    ind = sample_dict[sample]["meas"].index(meas)
    params = sample_dict[sample]

    data_pth = os.path.join(pth, "Data", params["pth"][ind], params["dir"][ind])
    img_name = params["pth"][ind][0:-1] + params["meas"][ind]
    return params, data_pth, img_name
