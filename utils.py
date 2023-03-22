import yaml

def load_config(configfile):
    # load config.yaml from current directory
    with open(configfile) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        # nested dictionary structure
        config = {}
        for k, v in conf.items():
            for k2, v2 in v.items():
                config[k2] = v2
    return config 
