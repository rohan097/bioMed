import json
import os

new_dict = {}
#new_dict['parameters'] = {'fs': 399.6097561, 'noverlap': 512, 'NFFT': 1024}
#new_dict['output_path'] = '/home/rohan/bioMed/'
if os.path.isfile('config.json'):
    with open('config.json', 'r') as f:
        config = json.load(f)

    if len(new_dict):
        config.update(new_dict)
        with open('config.json', 'w') as f:
            json.dump(config, f)

    print ("Updated 'Config.json': \n", config)
else:
    config = {'path' : '/media/rohan/My Stuff/EEG/'}
    with open('config.json', 'w') as f:
        json.dump(config, f)