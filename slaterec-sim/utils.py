import yaml
import names
def load_param(**kwargs):
    param = yaml.safe_load(open("config.yml", 'r'))
    # Replace parameters

    # generate random name
    param['name'] = f"{names.get_full_name().replace(' ','-')}"
    for key, val in kwargs.items():
        param[key] = val
    return param