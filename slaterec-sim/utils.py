import yaml
def load_param(**kwargs):
    param = yaml.safe_load(open("config.yml", 'r'))
    # Replace parameters
    for key, val in kwargs.items():
        param[key] = val
    return param