import json
import os

new_dict = {}

if os.path.isfile('config.json'):
    with open('config.json', 'r') as f:
        config = json.load(f)

    if not len(new_dict):
        config.update(new_dict)
        with open('config.json', 'w') as f:
            json.dump(config, f)

else:
    config = {'path' : '/media/rohan/My Stuff/EEG/'}
    with open('config.json', 'w') as f:
        json.dump(config, f)