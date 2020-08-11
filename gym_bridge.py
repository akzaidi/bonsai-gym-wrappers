#!/usr/bin/env python3
"""
MSFT Bonsai SDK3 Template for Simulator Integration using Python
Copyright 2020 Microsoft

Usage:
  For registering simulator with the Bonsai service for training:
    python simulator_integration.py --api-host https://api.bons.ai \
           --workspace <workspace_id> \
           --accesskey="<access_key> \
  Then connect your registered simulator to a Brain via UI
"""

import os
import pathlib
import sys
from dotenv import load_dotenv, set_key
import time
import datetime
from typing import Dict, Any, List
import gym
import wandb
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorState,
    SimulatorInterface,
)

dir_path = os.path.dirname(os.path.realpath(__file__))
brain_name = "gym-pole"
default_config = {"length": 1.5, "masspole": 0.1}


class TemplateSimulatorSession:
    def __init__(
        self,
        env_name: str = "CartPole-v1",
        render: bool = False,
        save_prefix: str = brain_name,
        save_runs: bool = False,
        log_file: str = None,
    ):
        """Template for simulating sessions with

        Parameters
        ----------
        env_name : str, optional
            [description], by default "Cartpole-v1"
        """

        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env_type = type(self.env)
        obs_dict = self.env.reset()
        self.observation = obs_dict
        self.render = render
        self.terminal = False
        self.save_prefix = save_prefix
        self.save_path = (
            f"videos/{self.save_prefix}__{self.env_name}__{int(time.time())}"
        )
        if not log_file:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file = current_time + "_" + env_name + "_log.csv"
            log_file = os.path.join("log", log_file)
            logs_directory = pathlib.Path(log_file).parent.absolute()
            if not pathlib.Path(logs_directory).exists():
                print(
                    "Directory does not exist at {0}, creating now...".format(
                        str(logs_directory)
                    )
                )
                logs_directory.mkdir(parents=True, exist_ok=True)
        self.log_file = log_file
        if save_runs:
            self.episode_save()
            wandb.init(
                project="bonsai-experiments",
                sync_tensorboard=False,
                name=self.env_name,
                monitor_gym=True,
            )
        self.sparse_reward = 0
        self.save_runs = save_runs

    def get_state(self) -> Dict[str, List]:
        """ Called to retreive the current state of the simulator. """

        return {
            "position": self.observation.tolist()[0],
            "velocity": self.observation.tolist()[1],
            "angle": self.observation.tolist()[2],
            "angular_velocity": self.observation.tolist()[3],
            "gym_terminal": int(self.terminal),
            "sparse_reward": self.sparse_reward,
        }

    def episode_start(self, config: Dict[str, Any] = config):
        """Method invoked at the start of each episode with a given 
        episode configuration.

        Parameters
        ----------
        config : Dict[str, Any]
        SimConfig parameters for the current episode defined in Inkling
        """

        self.observation = self.env.reset()
        if "length" in config.keys():
            self.env.env.length = config["length"]
            if self.save_runs:
                self.env.env.env.length = config["length"]
        if "masspole" in config.keys():
            self.env.env.masspole = config["masspole"]
            if self.save_video:
                self.env.env.env.masspole = config["masspole"]
        if "gravity" in config.keys():
            self.env.env.gravity = config["gravity"]
            if self.save_video:
                self.env.env.env.gravity = config["gravity"]
        if self.render:
            self.sim_render()
        if config is None:
            config = default_config
        self.config = config

    def episode_step(self, action: Dict[str, Any]):
        """Called for each step of the episode 

        Parameters
        ----------
        action : Dict[str, Any]
        BrainAction chosen from the Bonsai Service, prediction or exploration
        """

        self.observation, self.sparse_reward, self.terminal, self.info = self.env.step(
            action
        )
        if self.render:
            self.sim_render()
        if self.save_runs:
            wandb.log({**action, **self.observation, **self.config})

    def sim_render(self):

        if self.render:
            self.env.render()

    def halted(self) -> bool:
        """ Should return True if the simulator cannot continue"""
        return self.terminal

    def random_policy(self) -> Dict:

        return self.env.action_space.sample()

    def episode_save(self):

        self.env = Monitor(self.env, directory=self.save_path, resume=True)

    def log_iterations(self, state, action, episode: int = 0, iteration: int = 1):
        """Log iterations during training to a CSV.

        Parameters
        ----------
        state : Dict
        action : Dict
        episode : int, optional
        iteration : int, optional
        """

        import pandas as pd

        def add_prefixes(d, prefix: str):
            return {f"{prefix}_{k}": v for k, v in d.items()}

        state = add_prefixes(state, "state")
        action = add_prefixes(action, "action")
        config = add_prefixes(self.config, "config")
        data = {**state, **action, **config}
        data["episode"] = episode
        data["iteration"] = iteration
        log_df = pd.DataFrame(data, index=[0])

        if os.path.exists(self.log_file):
            log_df.to_csv(
                path_or_buf=self.log_file, mode="a", header=False, index=False
            )
        else:
            log_df.to_csv(path_or_buf=self.log_file, mode="w", header=True, index=False)


def env_setup():
    """Helper function to setup connection with Project Bonsai

    Returns
    -------
    Tuple
        workspace, and access_key
    """

    load_dotenv(verbose=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    env_file_exists = os.path.exists(".env")
    if not env_file_exists:
        open(".env", "a").close()

    if not all([env_file_exists, workspace]):
        workspace = input("Please enter your workspace id: ")
        set_key(".env", "SIM_WORKSPACE", workspace)
    if not all([env_file_exists, access_key]):
        access_key = input("Please enter your access key: ")
        set_key(".env", "SIM_ACCESS_KEY", access_key)

    load_dotenv(verbose=True, override=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    return workspace, access_key


def test_random_policy(
    render: bool = True, num_episodes: int = 10, save_video: bool = False
):

    sim = TemplateSimulatorSession(save_runs=True)
    for episode in range(num_episodes):
        iteration = 0
        terminal = False
        obs = sim.episode_start(config=default_config)
        action = sim.random_policy()
        while not terminal:
            sim.episode_step(action)
            obs = sim.observation
            terminal = sim.terminal
            reward = sim.sparse_reward
            iteration += 1
            print(
                f"Episode #{episode} Iteration #{iteration}: Reward = {reward}; terminal = {terminal}; obs = {obs}"
            )
            if render:
                sim.env.render()
    sim.env.close()


def main(
    render: bool = False,
    log_iterations: bool = False,
    config_setup: bool = False,
    save_runs: bool = False,
):
    """Main entrypoint for running simulator connections

    Parameters
    ----------
    render : bool, optional
        visualize steps in environment, by default True, by default False
    log_iterations: bool, optional
        log iterations during training to a CSV file
    """

    # workspace environment variables
    if config_setup:
        env_setup()
        load_dotenv(verbose=True, override=True)

    # Grab standardized way to interact with sim API
    sim = TemplateSimulatorSession(render=render, save_runs=save_runs)

    # Configure client to interact with Bonsai service
    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    # # Load json file as simulator integration config type file
    # with open("interface.json") as file:
    # interface = json.load(file)

    # Create simulator session and init sequence id
    registration_info = SimulatorInterface(
        name=sim.env_name,
        timeout=60,
        simulator_context=config_client.simulator_context,
    )
    registered_session = client.session.create(
        workspace_name=config_client.workspace, body=registration_info
    )
    print("Registered simulator.")
    sequence_id = 1

    try:
        while True:
            # Advance by the new state depending on the event type
            sim_state = SimulatorState(
                sequence_id=sequence_id, state=sim.get_state(), halted=sim.halted(),
            )
            event = client.session.advance(
                workspace_name=config_client.workspace,
                session_id=registered_session.session_id,
                body=sim_state,
            )
            sequence_id = event.sequence_id
            print("[{}] Last Event: {}".format(time.strftime("%H:%M:%S"), event.type))

            # Event loop
            if event.type == "Idle":
                time.sleep(event.idle.callback_time)
                print("Idling...")
            elif event.type == "EpisodeStart":
                sim.episode_start(event.episode_start.config)
                # sim.episode_start(config)
            elif event.type == "EpisodeStep":
                sim.episode_step(event.episode_step.action["command"])
            elif event.type == "EpisodeFinish":
                print("Episode Finishing...")
            elif event.type == "Unregister":
                client.session.delete(
                    workspace_name=config_client.workspace,
                    session_id=registered_session.session_id,
                )
                print("Unregistered simulator.")
            else:
                pass
    except KeyboardInterrupt:
        # Gracefully unregister with keyboard interrupt
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print("Unregistered simulator.")
    except Exception as err:
        # Gracefully unregister for any other exceptions
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print("Unregistered simulator because: {}".format(err))


if __name__ == "__main__":
    main(render=True, save_runs=False)
    # test_random_policy(render=False)
