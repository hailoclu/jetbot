import torch
import numpy as np

import carb

from pxr import UsdGeom, Gf, Sdf, Usd, PhysxSchema, PhysicsSchema, PhysicsSchemaTools, Semantics

import os
import time
import atexit
import asyncio
import numpy as np
import random
import matplotlib.pyplot as plt
import collections

from omni.isaac.synthetic_utils import visualization as vis
from omni.isaac.synthetic_utils import OmniKitHelper
from omni.isaac.synthetic_utils import SyntheticDataHelper

from jetbot import Jetbot
from road_environment import Environment

import gym
from gym import spaces


class JetbotEnv:
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, omni_kit, z_height=0, max_resets=10, updates_per_step=3, steps_per_rollout=1000, mirror_mode=False
    ):

        self.MIRROR_MODE = mirror_mode

        self.action_space = spaces.Box(low=0, high=2.0, shape=(2,), dtype=np.float32)
        # IMPORTANT NOTE!  SB3 wraps all image spaces in a transposer.
        # it assumes the image outputed is of standard form
        self.observation_space = spaces.Box(low=0, high=255, shape=(224, 224, 1), dtype=np.uint8)
        self.noise = 0.05

        # every time we update the stage, this is how much time will be simulated
        self.dt = 1 / 30.0
        self.omniverse_kit = omni_kit
        self.sd_helper = SyntheticDataHelper()
        self.roads = Environment(self.omniverse_kit)

        # make environment z up
        self.omniverse_kit.set_up_axis(UsdGeom.Tokens.z)

        # we are going to train on a randomized loop that fits in a 6x6 tile area.
        self.shape = [2, 2]
        self.roads.generate_road(self.shape)
        self.roads.generate_lights()

        # spawn robot
        self.jetbot = Jetbot(self.omniverse_kit)
        self.initial_loc = self.roads.get_valid_location()
        self.jetbot.spawn(Gf.Vec3d(self.initial_loc[0], self.initial_loc[1], 5), 0)

        # switch kit camera to jetbot camera
        self.jetbot.activate_camera()

        # start simulation
        self.omniverse_kit.play()

        # Step simulation so that objects fall to rest
        # wait until all materials are loaded
        frame = 0
        print("simulating physics...")
        while frame < 60 or self.omniverse_kit.is_loading():
            self.omniverse_kit.update(self.dt)
            frame = frame + 1
        print("done after frame: ", frame)

        self.initialized = False
        self.numsteps = 0
        self.numresets = 0
        self.maxresets = max_resets
        self.updates_per_step = updates_per_step
        self.steps_per_rollout = steps_per_rollout

        self.hist_length = collections.deque([0.0] * 10, maxlen=10)
        self.hist_forward_vel = collections.deque([0.0] * 10, maxlen=10)
        self.hist_ang_vel = collections.deque([0.0] * 30, maxlen=30)
        self.avg_forward_vel = 0
        self.dist_traveled = 0
        self.total_reward = 0

        # Randomly mirror horizontally
        self.update_mirror_mode()

    def update_mirror_mode(self):
        # Mirror if mode is enabled and we randomly sample True
        self.mirror_mode = self.MIRROR_MODE & random.choice([False, True])

    def calculate_reward(self):
        # distance to nearest point on path in units of block.  [0,1]
        dist = self.roads.distance_to_path_in_tiles(self.current_pose)
        self.dist = dist
        dist_reward = np.exp(-dist ** 2 / 0.15 ** 2)
        reward = self.current_forward_velocity * dist_reward
        # if we are driving backwards, large negative reward
        # if self.current_forward_velocity < 0:
        #     reward = self.current_forward_velocity

        # THIS IS FOR DEBUGGING ONLY
        if self.numsteps % 10 == 0 or reward < 0:
            print(
                "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                    reward,
                    self.current_forward_velocity,
                    self.avg_forward_vel,
                    dist_reward,
                    self.dist_traveled,
                    self.current_ang_vel,
                )
            )
        self.total_reward += reward
        return reward

    def is_dead(self):
        done = False
        # terminate if we leave boundary
        if not self.roads.is_inside_path_boundary(self.current_pose):
            print("dead not inside boundary", self.numsteps)
            done = True

        # kill the episode after 500 steps
        if self.numsteps > self.steps_per_rollout:
            print("dead self.numsteps > self.steps_per_rollout", self.numsteps)
            done = True

        # terminate if we are driving backwards for too long
        if self.avg_forward_vel <= 0 and self.numsteps > 35:
            print("dead self.avg_forward_vel <= 1 after 35 steps ", self.avg_forward_vel)
            done = True

        return done

    def transform_action(self, action):

        # If mirrored, swap wheel controls
        if self.mirror_mode:
            action = action[::-1]

        return action

    def transform_state_image(self, im):

        # If enabled, mirror image horizontally
        if self.mirror_mode:
            return np.flip(im, axis=1)

        return im

    def step(self, action):
        if self.initialized:
            self.previous_loc = self.current_loc

        transformed_action = self.transform_action(action)
        self.jetbot.command(transformed_action)
        frame = 0
        reward = 0

        # every time step is called we actually update the scene by updates_per_step.
        while frame < self.updates_per_step:
            # render at 1/30, simulate at 1/60, which means 2 substeps per frame
            self.omniverse_kit.update(self.dt, 1.0 / 60.0, 2.0)
            frame = frame + 1

        # compute reward once simulation is complete
        obs = self.jetbot.observations()
        self.current_pose = obs["pose"]
        self.current_speed = np.linalg.norm(np.array(obs["linear_velocity"]))
        self.current_forward_velocity = obs["local_linear_velocity"][0]
        self.current_ang_vel = obs["angular_velocity"][2]
        self.current_loc = self.roads.get_tile_from_pose(self.current_pose)
        self.hist_forward_vel.append(self.current_forward_velocity)
        self.dist_traveled = self.dist_traveled + self.current_forward_velocity * self.dt
        self.hist_ang_vel.append(self.current_ang_vel)
        self.avg_forward_vel = sum(self.hist_forward_vel) / len(self.hist_forward_vel)

        if not self.initialized:
            self.previous_loc = self.roads.get_tile_from_pose(self.current_pose)

        reward = self.calculate_reward()

        # the synthetic data helper is our way of grabbing the image data we need from the camera.  currently the SD helper
        # only supports a single camera, however you can use it to access camera data as a cuda tensor directly on the
        # device.  stable baselines 3 is expecting a numpy array, so we pull the data to the host
        # additional sensors that could be of interest and can be added to this list:
        # "depth", "instanceSegmentation", "semanticSegmentation"
        gt = self.sd_helper.get_groundtruth(["rgb", "camera"])

        # we only need the rgb channels of the rgb image
        currentState = gt["rgb"][:, :, :3].astype(np.float)
        currentState = self.transform_state_image(currentState)

        if not self.initialized:
            self.previousState = currentState

        img = np.dot(currentState, [0.299, 0.587, 0.114])  # np.concatenate((currentState, self.previousState), axis=2)
        img = img.reshape((img.shape[0], img.shape[1], 1))

        # the real camera will have noise on each pixel, so we add some uniform noise here to simulate thats
        # uncomment below to add noise to image
        img = np.clip((255 * self.noise * np.random.randn(224, 224, 1) + img.astype(np.float)), 0, 255).astype(np.uint8)

        self.previousState = currentState

        self.numsteps += 1
        done = self.is_dead()

        return img, reward, done, {}

    def reset(self):

        # Randomly mirror horizontally
        self.update_mirror_mode()

        # randomize the road configuration every self.maxresets resets.
        if self.numresets % self.maxresets == 0:
            size = random.randrange(2, 6)
            self.shape = [size, size]
            self.roads.reset(self.shape)

        if not self.initialized:
            state, reward, done, info, = self.step([0, 0])
            self.initialized = True

        # every time we reset, we move the robot to a random location, and pointing along the direction of the road
        loc = self.roads.get_valid_location()
        # the random angle offset can be increased here
        rot = self.roads.get_forward_direction(loc) + random.uniform(-10, 10)
        self.jetbot.teleport(
            Gf.Vec3d(loc[0] + random.uniform(-2.5, 2.5), loc[1] + random.uniform(-2.5, 2.5), 5), rot, settle=True
        )

        obs = self.jetbot.observations()
        self.current_pose = obs["pose"]
        self.current_speed = np.linalg.norm(np.array(obs["linear_velocity"]))
        self.current_forward_velocity = obs["local_linear_velocity"][0]
        self.current_loc = self.roads.get_tile_from_pose(self.current_pose)
        self.previous_loc = self.roads.get_tile_from_pose(self.current_pose)
        self.dist = self.roads.distance_to_path_in_tiles(self.current_pose)

        # wait for loading
        if self.numresets % self.maxresets == 0:
            while self.omniverse_kit.is_loading():
                self.omniverse_kit.update(self.dt)

        gt = self.sd_helper.get_groundtruth(["rgb", "camera"])
        currentState = gt["rgb"][:, :, :3]
        currentState = self.transform_state_image(currentState)
        img = np.dot(
            currentState.astype(np.float), [0.299, 0.587, 0.114]
        )  # np.concatenate((currentState, currentState), axis=2)
        img = img.reshape((img.shape[0], img.shape[1], 1))
        # uncomment below to add noise to image
        img = np.clip((255 * self.noise * np.random.randn(224, 224, 1) + img.astype(np.float)), 0, 255).astype(np.uint8)

        print(
            "reset ",
            sum(self.hist_length) / len(self.hist_length),
            self.numresets,
            self.dist_traveled,
            self.avg_forward_vel,
            self.total_reward,
        )

        self.numsteps = 0
        self.previousState = currentState
        self.numresets += 1
        self.total_reward = 0
        self.dist_traveled = 0

        return img
