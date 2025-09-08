import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def load_resonator_csv_combo(data_pth, cfg):
    fig, ax = plt.subplots(2, 4, figsize=(15, 7))
    ax = ax.flatten()
    df_full = pd.DataFrame()
    csv_files_in_dir = [
        f
        for f in os.listdir(os.path.join(data_pth, cfg["dir"][0]))
        if f.endswith(".csv") and f.startswith("fit_results")
    ]
    csv_files_in_dir.sort()
    print(csv_files_in_dir)
    coef = [1, 1, 1, 1, 1, 1]
    for i, csv_file in enumerate(csv_files_in_dir):
        for j in range(len(cfg["dir"])):
            file_path = os.path.join(data_pth, cfg["dir"][j], csv_file)
            df = pd.DataFrame()
            photon_max = []

            df_tmp = pd.read_csv(file_path)
            df_tmp["ind"] = j

            # This part is for combining data from different sweeps (i.e. adding an attenuator)
            if j > 0:
                df_tmp["photon_number"] = df_tmp["photon_number"] * 10 ** (
                    -coef[i] / 20
                )  # Adjust photon number

            df = pd.concat([df, df_tmp], ignore_index=True)
            # try:
            #     ax[i].semilogx(df_tmp["photon_number"], df_tmp["q_internal"], ".")
            #     ax[i].set_title(f'Pitch {cfg["pitch"][i]}')
            # except:
            #     pass
            photon_max.append(df_tmp["photon_number"].max())
        power_order = np.argsort(photon_max)
        df = df[
            (df["ind"] == power_order[0])
            | (df["photon_number"] >= photon_max[power_order[0]])
        ]

        df["pitch"] = cfg["pitch"][i]
        df["target_freq"] = cfg["target_freq"][i]
        df["resonator_id"] = i
        df["temp"] = 0.04
        df_full = pd.concat([df_full, df], ignore_index=True)

    df_full = df_full.sort_values(by="pitch")
    plot_raw(df_full)

    return df_full


def load_resonator_csv(data_pth, cfg, xval="photon_number"):
    # fig, ax = plt.subplots(2, 4, figsize=(15, 7))
    # ax = ax.flatten()
    df_full = pd.DataFrame()
    csv_files_in_dir = [
        f
        for f in os.listdir(os.path.join(data_pth, cfg["dir"][0]))
        if f.endswith(".csv") and f.startswith("fit_results")
    ]
    csv_files_in_dir.sort()
    print(csv_files_in_dir)
    for i, csv_file in enumerate(csv_files_in_dir):

        file_path = os.path.join(data_pth, cfg["dir"][0], csv_file)
        df = pd.read_csv(file_path)

        # try:
        #     ax[i].semilogx(df[xval], df["q_internal"], ".")
        #     ax[i].set_title(f'Pitch {cfg["pitch"][i]}')
        # except:
        #     pass

        df["pitch"] = cfg["pitch"][i]
        df["target_freq"] = cfg["target_freq"][i]
        df["resonator_id"] = i
        df["temp"] = 0.04
        df_full = pd.concat([df_full, df], ignore_index=True)

    df_full = df_full.sort_values(by="pitch")
    plot_raw(df_full, xval)

    return df_full


def plot_raw(df, xval):
    fig, ax = plt.subplots(2, 4, figsize=(15, 7))
    ax = ax.flatten()
    grouped = df.groupby("pitch")
    for i, (pitch, group) in enumerate(grouped):
        ax[i].plot(
            group[xval],
            group["q_internal"],
            "o",
            label=f"Gap: {pitch} µm \n Freq: {np.mean(group['frequency_Hz'])/1e9:.2f} GHz",
        )
        # ax[i].set_title(f'Pitch {pitch}')
        # ax[i].set_xlabel("Photon Number")
        ax[i].set_xlabel(xval)
        ax[i].set_ylabel("$Q_i$")
        if xval == "photon_number":
            ax[i].set_xscale("log")
        ax[i].legend()
    fig.tight_layout()
