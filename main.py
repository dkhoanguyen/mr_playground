#!/usr/bin/python3

import os
import yaml
import argparse

from mrs_playground.entity.animal import Animal
from mrs_playground.entity.robot import Robot

from mrs_playground.dynamic.point_mass_system import *
from mrs_playground.sensing.radius_sensing import RadiusSensing

from mrs_playground.behavior.mathematical_flock import MathematicalFlock

from mrs_playground.environment.simple_playground import SimplePlayground
from mrs_playground.environment.playground_factory import PlaygroundFactory

from mr_herding.behavior.distributed_outmost_push import DistributedOutmostPush

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(PROJECT_DIR, 'config')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gui',
                        help='Render gui', action='store_true')
    args = parser.parse_args()

    entity_config_file = os.path.join(CONFIG_DIR, 'entity_outmost.yaml')
    with open(entity_config_file, 'r') as file:
        entity_config = yaml.safe_load(file)

    env_config_file = os.path.join(CONFIG_DIR, 'playground.yaml')
    with open(env_config_file, 'r') as file:
        env_config = yaml.safe_load(file)

    behaviour_config_file = os.path.join(CONFIG_DIR, 'behavior.yaml')
    with open(behaviour_config_file, 'r') as file:
        behavior_config = yaml.safe_load(file)

    sensing_config_file = os.path.join(CONFIG_DIR, 'sensing_model.yaml')
    with open(sensing_config_file, 'r') as file:
        sensing_config = yaml.safe_load(file)

    dynamic_config_file = os.path.join(CONFIG_DIR, 'dynamic_model.yaml')
    with open(dynamic_config_file, 'r') as file:
        dynamic_config = yaml.safe_load(file)

    # Spawn animals
    animals = PlaygroundFactory.spawn_entities(entity_config['animal'], Animal)

    # Animal flocking behavior
    math_flock_config = behavior_config['math_flock']
    math_flock = MathematicalFlock(**math_flock_config['params'])
    for animal in animals:
        math_flock.add_animal(animal)

    # Spawn robots
    robots = PlaygroundFactory.spawn_entities(entity_config['robot'], Robot)
    # Add sensors and dynamics to robots
    sensors = PlaygroundFactory.add_sensing(entities=robots,
                                            config=sensing_config,
                                            sensing_type=RadiusSensing)
    PlaygroundFactory.add_dynamic(entities=robots,
                                  config=dynamic_config,
                                  dynamic_type=SingleIntegrator)
    # Add behavior as well
    behavior_config['outmost_push']['params'].update(
        {"sensing_range": sensing_config["sensing_radius"]})

    outmost_push_config = behavior_config['outmost_push']['params']
    PlaygroundFactory.add_behavior(entities=robots,
                                   config=behavior_config['outmost_push']['params'],
                                   behavior_type=DistributedOutmostPush,
                                   behavior_name="outmost")

    # Add robots
    for robot in robots:
        math_flock.add_robot(robot)

    # Create environment
    env = SimplePlayground(**env_config)

    # Add entities to env
    for animal in animals:
        env.add_entity(animal)
    for robot in robots:
        env.add_entity(robot)

    env.add_behaviour(math_flock)

    # Add sensor
    for sensor in sensors:
        env.add_sensing_models(sensor)

    env.turn_on_render(args.gui)

    while env.ok:
        env.step()
        env.render()

if __name__ == '__main__':
    main()
