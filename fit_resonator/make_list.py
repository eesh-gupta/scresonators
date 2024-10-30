

sample_dict = {'no hf':['240725'], 'hf':['240719', '240729', '240729-2', '240729-3', '240729-6'], 'hf-soc0':['240729-5','240729-6'], 'scalinq':['240814-2'], 'silicon-scalinq':['silicon-scalinq-1']}

# samples: no hf, hf, scalinq (0729-2 is hf3)
# meas: 240725, 240719, 240729, 240729-2, 240814-2

meas = 'silicon-scalinq-1'

matching_keys = [key for key, items in sample_dict.items() for item in items if item == meas]
sample = matching_keys[0] if matching_keys else None

full_dict = {}
for sample, value in sample_dict.items():
    full_dict[sample]= {'meas':value}
    
    if sample=='no hf':
        pth = '240725/'
        params = {'atten': -40, 'min_power': -100, 'max_power': -15, 'flip': False}
    elif sample=='hf':
        pth = '240729/'
        params = {'atten':-50, 'min_power': -100, 'max_power': -25, 'flip': True}
    elif sample=='hf-soc':
        pth = '240818-Resonator/'
        params = {'atten':-50, 'min_power': -100, 'max_power': 25, 'flip': True}
    elif sample=='hf-soc0':
        pth = '240816-Resonator/'
        params = {'atten':-50, 'min_power': -100, 'max_power': 25, 'flip': True}
    elif sample=='scalinq':
        pth = '240814-Resonator/'
        params = {'atten':-65, 'min_power': -100, 'max_power': -15, 'flip': False}    
    elif sample=='silicon-scalinq':    
        pth = '240908-Resonator/'
        params = {'atten':-65, 'min_power': -100, 'max_power': -15, 'flip': True}
    else:
        print('Sample not found')

    if params['flip']: 
        params['pitch'] = [10, 12, 14, 2, 16, 4, 6, 8]
        params['target_freq'] =[5.9, 6.2, 6.5, 7, 6.7, 7.2, 7.5, 7.9]
    else: 
        params['pitch'] = [10, 12, 14, 16, 2, 4, 6, 8]
        params['target_freq'] =[5.9, 6.2, 6.5, 6.7, 7, 7.2, 7.5, 7.9]
    for k, v in params.items():
        full_dict[sample][k] = v
    
    full_dict[sample]['pth'] = pth

    full_dict[sample]['dir']=[]
    full_dict[sample]['nfiles']=[]
    full_dict[sample]['meas_type']=[]
    for m in full_dict[sample]['meas']:
        nfiles=3
        slope=0
        meas_type='vna'

        if m=='240725':
            directories = ['power_sweep7']
            nfiles = 2
        elif m=='240719':
            params['min_power']=-85
            directories = ['sample1_power_sweep_2']
            nfiles = 1
        elif m=='240729':
            directories = ['power_sweep2']
            params['max_power'] = -30
            nfiles = 2
        elif m=='240729-2':
            directories = ['power_sweep5']
            # there is an issue with power sweep 5 for the 6 um data 
        elif m=='240729-3':
            directories = ['power_sweep4']
        elif m=='240814-2':
            directories = ['powersweep7']
        elif m=='240729-5':
            meas_type='soc'
            slope = 23.9
            directories=['powersweep0']
        elif m=='240729-6':
            meas_type='soc'
            #slope = 23.45293930972242
            slope = 23.9
            directories=['powersweep3']
        elif m=='silicon-scalinq-1':
            directories=['powersweep8']
        else:
            print('Sample not found')
        full_dict[sample]['dir'].append(directories)
        full_dict[sample]['nfiles'].append(nfiles)
        full_dict[sample]['meas_type'].append(meas_type)

import yaml

with open('resonator_meas.yaml', 'w') as file:
    yaml.dump(full_dict, file)