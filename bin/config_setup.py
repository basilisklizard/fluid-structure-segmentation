import json

d = {}
d['paths'] = {
    'data': './training_data/set0/',
    'model': './model/model_example.pt'
}

d['opti'] = {
    'learning_rate': 0.00005,
    'weight_decay': 0,
    'num_epochs': 2000,
    'log_step': 20
}

enc = {}
enc['layers'] = [
    {'input': 1, 'output': 8, 'f_size': 3, 'stride': 1, 'padding': 1},
    {'input': 8, 'output': 16, 'f_size': 3, 'stride': 1, 'padding': 1},
    {'input': 16, 'output': 32, 'f_size': 3, 'stride': 1, 'padding': 1}
]

dec = {}
dec['layers'] = [
    {'input': 32, 'output': 16, 'f_size': 3, 'stride': 1, 'padding': 1},
    {'input': 16, 'output': 8, 'f_size': 3, 'stride': 1, 'padding': 1},
    {'input': 8, 'output': 1, 'f_size': 3, 'stride': 1, 'padding': 1}
]

d['network'] = {
    'type': 'u-net',
    'encoder': enc,
    'decoder': dec
}

with open('config_example.json', 'w') as file:
    json.dump(d, file, indent=4)
