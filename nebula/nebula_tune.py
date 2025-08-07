import multiprocessing
from typing import Generator
import nebula.core
import gc
import traceback


class space:

    @staticmethod
    def int(name_:str, min_:int, max_:int) -> dict:
        return {name_: {"type": "int", "min": min_, "max": max_}}

    @staticmethod
    def float(name_:str, min_:float, max_:float) -> dict:
        return {name_: {"type": "float", "min": min_, "max": max_}}

    @staticmethod
    def cat(name_:str, options_:dict) -> dict:
        return {name_: {"type": "categorical", "options": options_}}

    @staticmethod
    def bool(name_:str) -> dict:
        return {name_: {"type": "categorical", "options": [True, False]}}

def genetic_config(population_size=30,elite_size=4,mutation_rate=0.35) -> dict:
    return {'population_size':population_size, 'elite_size':elite_size,'mutation_rate':mutation_rate}

def Tuner (
        param_space: dict | list,
        genetic_config: dict | type[genetic_config],
        param_class: dict | None,
        name: str,
        class_df: type,
        conf_path: str="conf.json",
        batch_size: int=1000,
        generations: int=40,
        results_path: str="log/nebula_tuning.csv",
        core_use:int=None,
        verbose:bool=True,
        early_stopping:bool=True
        ) -> Generator[dict, None, None]:
    try:
        if isinstance(param_space, list):
            param_space = {k: v for d in param_space for k, v in d.items()}
            
        total_cores = multiprocessing.cpu_count()
        if core_use is None or core_use>total_cores:
            core_use = total_cores
        
        if param_class == None:
            param_class = {}

        if 'n_jobs' in param_class and core_use>1:
            if param_class['n_jobs'] == -1 or param_class['n_jobs'] > 1:
                if verbose:
                    print(f"Adjusting core_use from {core_use} to 1 to prevent resource conflict")
                core_use = 1
        try:
            test_instance = class_df(**param_class)
            test_result = test_instance.run(**{k: v['min'] if 'min' in v else v['options'][0] 
                                          for k, v in param_space.items()})
            if not isinstance(test_result, (int, float)):
                raise ValueError("Strategy must return numeric score")

        except Exception as e:
            print(f"Strategy validation failed: {str(e)}")
            print("Please check:")
            print(f"- param_class keys: {list(param_class.keys())}")
            print(f"- param_space keys: {list(param_space.keys())}")
            return
        
        if verbose:
            print(f"Starting Nebula optimization for '{name}'")
            print(f"Parameter space: {len(param_space)} dimensions")
            print(f"Genetic config: {genetic_config}")
            print(early_stopping)

        tuner = nebula.core.NebulaTunerEngine(
            param_space=param_space,
            genetic_config=genetic_config,
            param_class=param_class,
            name=name,
            conf_path=conf_path,
            batch_size=batch_size,
            class_df=class_df,
            results_path =results_path,
            verbose=verbose,
            core_use=core_use
        )

        tuner.optimize(
            generations=generations,
            results_path=results_path,
            early_stopping=early_stopping
        )

        gc.collect()
        print("nebula done")
    except KeyboardInterrupt:
        print("Optimization interrupted by user")
    except ImportError as e:
        print(f"Critical import error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        if 'tuner' in locals() and hasattr(tuner, '_save_final'):
            print("Attempting to save partial results...")
            try:
                tuner._save_final(results_path, total_time="N/A (crashed)")
            except Exception as save_error:
                print(f"Failed to save partial results: {str(save_error)}")
    finally:
        gc.collect()
        if verbose:
            print("Resources cleaned up")
