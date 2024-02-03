import os
import torch

old_folder = "old"
new_folder = "new" #"185k_sigmoid"

# Config that serves all environment
GLOBAL_CONFIG = {
    "OLD": {
        "CE": f"../models/{old_folder}/content_encoder.pth.tar",
        "SE": f"../models/{old_folder}/style_encoder.pth.tar",
        "DEC": f"../models/{old_folder}/decoder.pth.tar",
        "MOD": f"../models/{old_folder}/modulator.pth.tar"
    },
    "NEW": {
        "CE": f"../models/{new_folder}/content_encoder.pth.tar",
        "SE": f"../models/{new_folder}/style_encoder.pth.tar",
        "DEC": f"../models/{new_folder}/decoder.pth.tar",
        "MOD": f"../models/{new_folder}/modulator.pth.tar"
    },
    "USE_CUDA_IF_AVAILABLE": True,
    "ROUND_DIGIT": 6
}

# Environment specific config, or overwrite of GLOBAL_CONFIG
ENV_CONFIG = {
    "development": {
        "DEBUG": True
    },

    "staging": {
        "DEBUG": True
    },

    "production": {
        "DEBUG": False,
        "ROUND_DIGIT": 3
    }
}


def get_config() -> dict:
    """
    Get config based on running environment
    :return: dict of config
    """

    # Determine running environment
    ENV = os.environ['PYTHON_ENV'] if 'PYTHON_ENV' in os.environ else 'development'
    ENV = ENV or 'development'

    # raise error if environment is not expected
    if not ENV in ENV_CONFIG.keys():
        raise EnvironmentError(f'Config for envirnoment {ENV} not found')

    config = GLOBAL_CONFIG.copy()
    config.update(ENV_CONFIG[ENV])

    config['ENV'] = ENV
    config['DEVICE'] = 'cuda' if torch.cuda.is_available() and config['USE_CUDA_IF_AVAILABLE'] else 'cpu'

    return config

# load config for import
CONFIG = get_config()

if __name__ == '__main__':
    # for debugging
    import json
    print(json.dumps(CONFIG, indent=4))
