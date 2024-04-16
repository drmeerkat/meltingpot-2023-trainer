import os
import ray
import torch
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms import ppo
from ray.tune import registry
from environments.al_harvest_env import al_harvest_env_creator
from config import get_experiment_config
from datetime import datetime


# USER INPUT
timestamp = datetime.now().strftime("%Y%m%d-%H%M-%S")
# if args.debug:
#     args.base_log_dir = os.path.join(args.base_log_dir, f"{timestamp}-debug-logs")
# elif args.eva:
#     args.base_log_dir = os.path.join(args.base_log_dir, f"{timestamp}-eva-logs")
# else:
#     args.base_log_dir = os.path.join(args.base_log_dir, f"{timestamp}-logs")
# if not os.path.isdir(args.base_log_dir): 
#     os.makedirs(args.base_log_dir)
# TODO: do you need timestamp in ray log?

# This output_dir is the path used in your container!!
output_dir = f'/workspace/logs/{timestamp}-ray-logs'
num_workers = 4
use_tf_board = True
random_seed = 136838
# the below settings should only be changed if you add support for a new substrate
experiment_name = f'al_harvest'
substrate_name = 'allelopathic_harvest__open'
env_creator = al_harvest_env_creator



os.environ['RAY_memory_monitor_refresh_ms'] = '0'
ray.init(local_mode=False, ignore_reinit_error=True, num_gpus=int(torch.cuda.is_available()))
registry.register_env("meltingpot", env_creator)
default_config = ppo.PPOConfig()
default_config.use_custom_reward = False
experiment_name += '_base' if not default_config.use_custom_reward else ''
configs, exp_config, tune_config = get_experiment_config(default_config, 
                                                         output_dir, 
                                                         num_workers, 
                                                         experiment_name,
                                                         substrate_name,
                                                         env_creator)
# Set seed for deterministic training
configs.debugging(seed=random_seed)

# Enable gpu training, maybe you can limit this to 1
if torch.cuda.is_available():
    # configs.num_gpus = torch.cuda.device_count()
    configs.num_gpus = 1

if "WANDB_API_KEY" in os.environ:
    wandb_project = f'{experiment_name}_torch'
    wandb_group = "meltingpot"

    # Set up Weights And Biases logging if API key is set in environment variable.
    callbacks = [
        WandbLoggerCallback(
            project=wandb_project,
            group=wandb_group,
            api_key=os.environ["WANDB_API_KEY"],
            log_config=True,
        )
    ]
elif use_tf_board:
    # project_name = f'{experiment_name}'
    callbacks = [
    ] 
else:
    callbacks = []
    print("WARNING! No callbacks found, running without wandb or tfboard!")

ckpt_config = air.CheckpointConfig(num_to_keep=exp_config['keep'], 
                                   checkpoint_frequency=exp_config['freq'], 
                                   checkpoint_at_end=exp_config['end'], 
                                  )

tuner = tune.Tuner(
        'PPO',
        param_space=configs.to_dict(),
        run_config=air.RunConfig(name = exp_config['name'], callbacks=callbacks, storage_path=exp_config['dir'], 
                                stop=exp_config['stop'], checkpoint_config=ckpt_config, verbose=0),
    )

results = tuner.fit()

best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
print(best_result)

ray.shutdown()

# nohup docker compose -f "/home/meerkat/Developments/rewardshaping/docker-compose-rs.yml" run --rm -w /workspace/code development python -u train.py > /home/meerkat/Developments/rewardshaping/rs.log 2>&1 &
