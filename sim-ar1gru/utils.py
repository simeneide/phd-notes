import yaml
import names
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')


def load_param(**kwargs):
    param = yaml.safe_load(open("config.yml", 'r'))
    # Replace parameters

    # generate random name
    random_name = f"{names.get_full_name().replace(' ','-')}"
    param['name'] = f"{random_name}-{param.get('name')}"
    for key, val in kwargs.items():
        param[key] = val

    logging.info("--------")
    logging.info(f"--- LOADED MODEL {param['name']}")
    logging.info("--------")
    return param