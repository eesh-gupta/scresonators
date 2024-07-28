import os 
import numpy as np
import matplotlib.pyplot as plt
import time 
import re
import fit_resonator.resonator as scres
import handy as hy 
from collections import Counter


def get_resonators(folder, pth, pattern):
    
    # List of files
    pth = pth + folder
    file_list0 = os.listdir(pth)
    #print(len(file_list0))

    tokens=[]
    # Get a list of resonators 
    for i in range(len(file_list0)):
        tokens.append(re.findall(pattern, file_list0[i]))

    values = [int(token) for sublist in tokens for token in sublist]
    resonators = set(values)

    frequency = Counter(values)
    print(frequency)

    resonators = np.array(list(resonators))
    resonators.sort()
    return resonators, file_list0

def get_resonator_power_list(pattern, file_list0):
    # Grab all the files for a given resonator, then sort by power.                         
    file_list = [file for file in file_list0 if re.match(pattern, file)]
    tokens=[]
    for i in range(len(file_list)):
        tokens.append(re.findall(pattern, file_list[i])[0])
    powers = np.array(tokens, dtype=float)
    inds = np.argsort(powers)
    
    file_list = [file_list[i] for i in inds]
    file_list = file_list[::-1]
    print(file_list)
    return file_list

def fit_resonator(data, filename, output_path, fit_type='DCM', plot=None, pre='circle'):
    # fit type DCM, CPZM
    my_resonator = scres.Resonator()
    my_resonator.outputpath = output_path 
    my_resonator.filename = filename
    my_resonator.from_columns(data['freqs'], data['amps'], data['phases'])
    # Set fit parameters

    MC_iteration = 4
    MC_rounds = 1e3
    MC_fix = []
    manual_init = None
    my_resonator.preprocess_method = pre # Preprocess method: default = linear
    my_resonator.filepath = './' # Path to fit output
    # Perform a fit on the data with given parameters
    my_resonator.fit_method(fit_type, MC_iteration, MC_rounds=MC_rounds, MC_fix=MC_fix,manual_init=manual_init, MC_step_const=0.3)
    output = my_resonator.fit(plot)
    return output 

# conf_array = [Q_conf, Qi_conf, Qc_conf, Qc_Re_conf, phi_conf, w1_conf]
# Q, Qc, Frequency, Phase 

def check_phase(data):
    if (np.max(data)-np.min(data))>np.pi: 
       data = data*np.pi/180
    return data  

def grab_data(pth, fname):
    data, attrs = hy.prev_data(pth, fname)

    # Reformat data for scres package
    data['phases'] = np.unwrap(data['phases'][0])
    data['phases'] = check_phase(data['phases'])
    data['freqs']=data['fpts'][0]
    data['amps']=data['mags'][0]
    return data

def plot_raw_data(data, phs_off = 0, amp_off = 0):
    fig, ax = plt.subplots(1,2, figsize=(10,4))    
    ax[0].plot(data['freqs']/1e9, data['phases']+phs_off)
    ax[1].plot(data['freqs']/1e9, data['amps']-amp_off)
    ax[0].set_xlabel('Frequency (GHz)')
    ax[0].set_ylabel('Phase (rad)')
    ax[1].set_xlabel('Frequency (GHz)')
    ax[1].set_ylabel('Amplitude (dB)')
    phs = data['phases']+phs_off
    amp = 10**((data['amps']-amp_off)/20)
    fig.tight_layout()
    plt.figure()

    plt.plot(amp*np.cos(phs), amp*np.sin(phs), '.')


def combine_data(data1, data2): 
    data = {}
    keys_list = ['freqs', 'amps', 'phases']
    for key in keys_list: 
        data[key] = np.concatenate([data1[key], data2[key]])
    
    inds = data['freqs'].argsort()
    for key in keys_list: 
        data[key] = data[key][inds]
    return data


def stow_data(params, res_params, j, power, err):
                            # Put all the data for a given resonator/temp in arrays. 
    power=np.array(power)            
    q = np.array([params[k][0] for k in range(len(params))])
    qc = np.array([params[k][1] for k in range(len(params))])
    freq = np.array([params[k][2] for k in range(len(params))])
    phase = np.array([params[k][3] for k in range(len(params))])
    qi_phi = 1/(1/q-1/qc)
    Qc_comp = qc / np.exp(1j * phase)
    Qi = (q ** -1 - np.real(Qc_comp ** -1)) ** -1
    
    res_params[j]['pow'].append(power)
    res_params[j]['q'].append(q)
    res_params[j]['qc'].append(qc)
    res_params[j]['freqs'].append(freq)
    res_params[j]['phs'].append(phase)
    res_params[j]['qi_phi'].append(qi_phi)
    res_params[j]['qi'].append(Qi)

    res_params[j]['phs_err'].append(np.array([err[i][4] for i in range(len(err))]))
    res_params[j]['q_err'].append(np.array([err[i][0] for i in range(len(err))]))
    res_params[j]['qi_err'].append(np.array([err[i][1] for i in range(len(err))]))
    res_params[j]['qc_err'].append(np.array([err[i][2] for i in range(len(err))]))
    res_params[j]['qc_real_err'].append(np.array([err[i][3] for i in range(len(err))]))
    res_params[j]['f_err'].append(np.array([err[i][5] for i in range(len(err))]))
    

    fig, ax = plt.subplots(2,1, figsize=(6,7.5), sharex=True)
    ax[0].plot(power, Qi, '.', markersize=6)
    ax[0].set_ylabel('$Q_i$')
    ax[1].plot(power, qc, '.', markersize=6)
    ax[1].set_ylabel('$Q_c$')
    ax[1].set_xlabel('Power (dBm)')
    fig.suptitle('$f_0 = $ {:3.3f} GHz'.format(freq[0]/1e9))
    fig.tight_layout()

    return res_params
    #plt.savefig('/Users/sph/' + 'resonator_power_'+str(directories[i]) + str(resonators[j]) + '.png', dpi=300)                 


def analyze_sweep(directories, pth_base, plot=None, min_power=-100):
    pattern0 = r'res_(\d+)_\d{2,3}dbm'

    # Initialize dict by getting list of resonators, creating dict with len = n resonators
    resonators, file_list = get_resonators(directories[0],pth_base, pattern0)
    res_params = [None] * len(resonators)
    for i in range(len(resonators)):
        res_params[i] = {'freqs':[], 'phs':[], 'q':[], 'qi':[], 'qc':[], 'qi_phi':[], 'pow':[], 'qi_err':[], 'q_err':[], 'phs_err':[], 'qc_err':[], 'qc_real_err':[], 'f_err':[]}
    
    # Each directory is a temperature 
    for i in range(len(directories)): 
        start = time.time()
        print(i)
        output_path = '../../../Images/res_name_' + directories[i] + '/'
        resonators, file_list0 = get_resonators(directories[i], pth_base, pattern0)
        pth = pth_base + directories[i]
        for j in range(len(resonators)): 
            # Grab all the files for a given resonator, then sort by power. 
            pattern = 'res_{:d}_'.format(resonators[j]) + '(\d{2,3})dbm'
            file_list = get_resonator_power_list(pattern, file_list0)
            
            params, err, power = [], [], []
            for k in range(len(file_list)):
                data = grab_data(pth, file_list[k])

                # Skip really noisy ones since they are slow
                if data['vna_power'][0] < min_power: 
                    continue
                power.append(data['vna_power'][0])
                
                try:                                            
                    output = fit_resonator(data, file_list[k], output_path, plot=plot)    
                    params.append(output[0])
                    err.append(output[1])
                except Exception as error:
                    print("An exception occurred:", error) 
                    params.append(np.nan*np.ones(4))
                    err.append(np.nan*np.ones(6))
            res_params = stow_data(params, res_params, j, power, err)
            
            print('Time elapsed: ', time.time()-start)            

    for i in range(len(resonators)):
        for key in res_params[i].keys():
            res_params[i][key] = np.array(res_params[i][key])

    return res_params


def analyze_sweep_double(directories, pth_base, plot=None, min_power=-100):
    pattern0 = r'res_(\d+)_\d{2,3}dbm_wide'

    # Initialize dict by getting list of resonators, creating dict with len = n resonators
    resonators, file_list = get_resonators(directories[0],pth_base, pattern0)
    res_params = [None] * len(resonators)
    for i in range(len(resonators)):
        res_params[i] = {'freqs':[], 'phs':[], 'q':[], 'qi':[], 'qc':[], 'qi_phi':[], 'pow':[], 'qi_err':[], 'q_err':[], 'phs_err':[], 'qc_err':[], 'qc_real_err':[], 'f_err':[]}
    
    # Each directory is a temperature 
    for i in range(len(directories)): 
        start = time.time()
        print(i)
        output_path = '../../../Images/res_name_' + directories[i] + '/'
        resonators, file_list0 = get_resonators(directories[i], pth_base, pattern0)
        pth = pth_base + directories[i]
        for j in range(len(resonators)): 
            # Grab all the files for a given resonator, then sort by power. 
            pattern = 'res_{:d}_'.format(resonators[j]) + '(\d{2,3})dbm_wide'
            file_list = get_resonator_power_list(pattern, file_list0)
            
            params, err, power = [], [], []
            for k in range(len(file_list)):
                data1 = grab_data(pth, file_list[k])
                file2 = file_list[k].replace('wide', 'narrow')
                try: 
                    data2 = grab_data(pth, file2)
                    data = combine_data(data1, data2)
                except:
                    continue

                # Skip really noisy ones since they are slow
                if data1['vna_power'][0] < min_power: 
                    continue
                power.append(data1['vna_power'][0])
                
                try:                                            
                    output = fit_resonator(data, file_list[k], output_path, plot=plot)    
                    params.append(output[0])
                    err.append(output[1])
                except Exception as error:
                    print("An exception occurred:", error) 
                    params.append(np.nan*np.ones(4))
                    err.append(np.nan*np.ones(6))
            res_params = stow_data(params, res_params, j, power, err)
            
            print('Time elapsed: ', time.time()-start)            

    for i in range(len(resonators)):
        for key in res_params[i].keys():
            res_params[i][key] = np.array(res_params[i][key])

    return res_params