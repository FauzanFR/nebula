from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os

import numpy as np


def strip_result_keys(individual, allowed_keys):
    return {k: v for k, v in individual.items() if k in allowed_keys}

import copy

def save_best_config(name, best_params, conf_path):
    try:
        # Create a clean copy first
        params_clean = copy.deepcopy(best_params)
        
        # Remove unwanted keys
        keys_to_remove = ['score', 'trial_time']
        for key in keys_to_remove:
            params_clean.pop(key, None)

        if os.path.exists(conf_path):
            with open(conf_path, "r") as f:
                all_confs = json.load(f)
        else:
            all_confs = []

        idx = next((i for i, c in enumerate(all_confs) if c["name"] == name), None)

        new_conf_entry = {
            "name": name,
            "version": datetime.now().strftime("%d-%m-%Y"),
            "conf": [params_clean]
        }

        if idx is not None:
            all_confs[idx] = new_conf_entry
        else:
            all_confs.append(new_conf_entry)

        with open(conf_path, "w") as f:
            json.dump(all_confs, f, indent=4)

        print(f"Configuration for '{name}' has been saved to: {conf_path}")

    except (IOError, json.JSONDecodeError) as e:
        print(f"Failed to save config: {str(e)}")
        temp_path = f"{conf_path}.tmp"
        with open(temp_path, "w") as f:
            json.dump(best_params, f)
        print(f"Saved temporary config to {temp_path}")


def setup_logger(name="nebula", log_level=logging.INFO, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    )

    file_handler = RotatingFileHandler(
        f"{log_dir}/nebula.log", 
        maxBytes=10*1024*1024, 
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    if not logger.handlers: 
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def normalize(individual, param_space):
    try:
        norm = []
        for param, config in param_space.items():
            val = individual.get(param, None)

            if config["type"] == "float":
                if val is None:
                    val = config.get("min", 0)
                norm.append((val - config["min"]) / (config["max"] - config["min"]))

            elif config["type"] == "int":
                if val is None:
                    val = config.get("min", 0)
                norm.append((val - config["min"]) / (config["max"] - config["min"]))

            elif config["type"] == "categorical":
                if val not in config["options"]:
                    val = config["options"][0]
                onehot = [0] * len(config["options"])
                onehot[config["options"].index(val)] = 1
                norm.extend(onehot)

            else:
                # fallback safe
                norm.append(0)
        return np.array(norm)
    except KeyError as e:
        print(f"Missing parameter in normalization: {str(e)}")
        # Return zero vector as fallback
        dim = sum(
            len(config['options']) if config['type'] == 'categorical' else 1 
            for config in param_space.values()
        )
        return np.zeros(dim)
    
class NebulaInitializationError(Exception):
    """Custom exception for strategy initialization failures"""
    pass