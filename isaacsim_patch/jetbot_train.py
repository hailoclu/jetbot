import torch
import numpy as np
import os
import carb
import signal
import json
import argparse
from argparse import Namespace

from omni.isaac.synthetic_utils import OmniKitHelper

from jetbot_env import JetbotEnv
from jetbot_model import CustomCNN

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


def train(args):
    CUSTOM_CONFIG = {
        "width": 224,
        "height": 224,
        "renderer": "RayTracedLighting",
        "headless": args.headless,
        "experience": f'{os.environ["EXP_PATH"]}/isaac-sim-python.json',
    }
    omniverse_kit = OmniKitHelper(CUSTOM_CONFIG)

    # we disable all anti aliasing in the render because we want to train on the raw camera image.
    omniverse_kit.set_setting("/rtx/post/aa/op", 0)

    env = JetbotEnv(omniverse_kit, max_resets=args.rand_freq, updates_per_step=3, mirror_mode=args.mirror_mode)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq, save_path="./params/", name_prefix=args.checkpoint_name
    )

    net_arch = [256, 256, dict(pi=[128, 64, 32], vf=[128, 64, 32])]
    policy_kwargs = {"net_arch": net_arch, "activation_fn": torch.nn.ReLU}

    if args.loaded_checkpoint == "":
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=args.tensorboard_dir,
            policy_kwargs=policy_kwargs,
            device="cuda",
            n_steps=args.step_freq,
            batch_size=2048,
            n_epochs=50,
            learning_rate=0.0001,
        )

    else:
        model = PPO.load(args.loaded_checkpoint, env)

    model.learn(
        total_timesteps=args.total_steps,
        callback=checkpoint_callback,
        eval_env=env,
        eval_freq=args.eval_freq,
        eval_log_path=args.evaluation_dir,
        reset_num_timesteps=args.reset_num_timesteps,
    )
    model.save(args.checkpoint_name)


def runEval(args):
    CUSTOM_CONFIG = {
        "width": 224,
        "height": 224,
        "renderer": "RayTracedLighting",
        "headless": args.headless,
        "experience": f'{os.environ["EXP_PATH"]}/isaac-sim-python.json',
    }
    # load a zip file to evaluate here
    agent = PPO.load(args.evaluation_dir + "/best_model.zip", device="cuda")

    omniverse_kit = OmniKitHelper(CUSTOM_CONFIG)

    # we disable all anti aliasing in the render because we want to train on the raw camera image.
    omniverse_kit.set_setting("/rtx/post/aa/op", 0)

    env = JetbotEnv(omniverse_kit, mirror_mode=args.mirror_mode)
    obs = env.reset()

    while True:
        action = agent.predict(obs)
        print(action)
        obs, rew, done, infos = env.step(action[0])
        if done:
            obs = env.reset()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "loaded_checkpoint",
        help="path to checkpoint to be loaded",
        default="/home/hmazhar/repos/omni_isaac_sim/source/python_samples/eval_log/best_model.zip",
        nargs="?",
        type=str,
    )

    parser.add_argument("-E", "--eval", help="evaluate checkpoint", action="store_true")

    parser.add_argument(
        "-R", "--reset_num_timesteps", help="reset the current timestep number (used in logging)", action="store_true"
    )

    parser.add_argument(
        "-M", "--mirror_mode", help="reflect images and actions horizontally during training", action="store_true"
    )

    parser.add_argument("-H", "--headless", help="run in headless mode (no GUI)", action="store_true")

    parser.add_argument(
        "--checkpoint_name", help="name of checkpoint file (no suffix)", default="checkpoint_25k", type=str
    )

    parser.add_argument("--tensorboard_dir", help="path to tensorboard log directory", default="tensorboard", type=str)

    parser.add_argument("--evaluation_dir", help="path to evaluation log directory", default="eval_log", type=str)

    parser.add_argument("--save_freq", help="number of steps before saving a checkpoint", default=4096 * 8, type=int)

    parser.add_argument("--eval_freq", help="number of steps before running an evaluation", default=4096 * 8, type=int)

    parser.add_argument("--step_freq", help="number of steps before executing a PPO update", default=10240, type=int)

    parser.add_argument(
        "--rand_freq", help="number of environment resets before domain randomization", default=1, type=int
    )

    parser.add_argument(
        "--total_steps",
        help="the total number of steps before exiting and saving a final checkpoint",
        default=250000000,
        type=int,
    )

    parser.add_argument(
        "--experimentFile", help="specify configuration via JSON.  Overrides commandline", default="", type=str
    )

    args = parser.parse_args()

    if args.experimentFile != "":
        args_dict = vars(args)
        if os.path.exists(args.experimentFile):
            with open(args.experimentFile) as f:
                json_args_dict = json.load(f)

                args_dict.update(json_args_dict)
                args = Namespace(**args_dict)

    print("running with args: ", args)

    def handle_exit(*args, **kwargs):
        print("Exiting training...")
        quit()

    signal.signal(signal.SIGINT, handle_exit)

    if args.eval:
        runEval(args)
    else:
        train(args)
