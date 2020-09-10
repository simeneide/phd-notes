import yaml
import names
import logging
import copy
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')


def load_param(**kwargs):
    param = yaml.safe_load(open("config.yml", 'r'))

    # generate random name
    random_name = f"{names.get_full_name().replace(' ','-')}"
    param['name'] = f"{random_name}-{param.get('name')}"
    for key, val in kwargs.items():
        param[key] = val

    logging.info("--------")
    logging.info(f"--- LOADED MODEL {param['name']}")
    logging.info("--------")
    return param

def load_sim_param(**kwargs):
    logging.info(f"--- LOADING SIMULATION PARS")
    param = load_param()
    # Build simulation parameters (copy inn default config if it doesnt exist in sim)
    simconfig = yaml.safe_load(open("config_simulation.yml","r"))
    # overwrite with simconfig:
    for key, val in simconfig.items():
        param[key] = val
    return param