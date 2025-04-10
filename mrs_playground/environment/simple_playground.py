#!/usr/bin/python3
import os
import cv2
import pickle
import time
from typing import Dict, List

import pygame
import numpy as np

from mrs_playground.params import params

from mrs_playground.common.entity import Entity
from mrs_playground.common.behavior import Behavior
from mrs_playground.common.sensing import SensingModel


class SimplePlayground(object):
    def __init__(self,
                 entity_names: list,
                 dt: float,
                 render: bool = True,
                 multi_threaded: bool = False,
                 save_to_file: bool = True,
                 save_path: str = "data/",
                 max_t: int = 3000):

        self._dt = dt
        self._multi_threaded = multi_threaded
        self._render = render
        self._save_to_file = save_to_file
        self._save_path = save_path
        self._max_t = max_t
        self._current_t = 0

        # Pygame for visualisation
        self._init = False
        self._running = True

        self._entities: Dict[str, List[Entity]] = {}
        self._behaviors: List[Behavior] = []
        self._sensing_model: List[SensingModel] = []

        self._entity_names: List[str] = entity_names

        for entity_name in self._entity_names:
            self._entities[entity_name] = []

        self._data_to_save = {}
        self._data_to_save["data"] = []

        # self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # self._video_writer = cv2.VideoWriter(
        #     "high_apf_1.mp4", self._fourcc, 200, params.SCREEN_SIZE)

        if self._render:
            if not self._init:
                pygame.init()
                self._screen = pygame.display.set_mode(params.SCREEN_SIZE)
                self._rect = self._screen.get_rect()
                self._clock = pygame.time.Clock()
                self._init = True

    def turn_on_render(self, status):
        self._render = status

    @property
    def ok(self):
        return self._running and self._current_t <= self._max_t

    def add_meta_data(self, meta_data):
        self._data_to_save["meta"] = meta_data

    def add_entity(self, entity: Entity):
        self._entities[entity.__str__()].append(entity)

    def add_behaviour(self, behavior: Behavior):
        self._behaviors.append(behavior)

    def add_sensing_models(self, sensing_model: SensingModel):
        self._sensing_model.append(sensing_model)

    def display(self):
        entity: Entity
        for entities in self._entities.values():
            for entity in entities:
                entity.display(self._screen)
        behavior: Behavior
        for behavior in self._behaviors:
            behavior.display(self._screen)
        

    def env_step(self):
        events = pygame.event.get()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

    def step(self):
        events = pygame.event.get()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
        behavior: Behavior
        for behavior in self._behaviors:
            behavior.update(events)
            if behavior._is_at_target:
                self._running = False

        # Grab all states
        all_states = {}
        for entity_type in self._entities.keys():
            all_states[entity_type] = np.empty((0, 6))
            for entity in self._entities[entity_type]:
                all_states[entity_type] = np.vstack(
                    (all_states[entity_type], entity.state))

        # Delegate all states to sensors, ie sensors are "sensing" the environment
        for sensing_model in self._sensing_model:
            sensing_model.update(all_states=all_states)

        # All comms
        all_comms = []
        if 'robot' in self._entity_names:
            for entity in self._entities['robot']:
                all_comms.append(entity.comms)

        # Update all entities
        for entity_type in self._entities.keys():
            for entity in self._entities[entity_type]:
                entity.update(events=events,
                              comms=all_comms)

        all_states.update({"ts": time.time()})
        self._data_to_save["data"].append(all_states.copy())
        self._current_t += 1

    def render(self, fps=60):
        if self._render:
            if not self._init:
                pygame.init()
                self._screen = pygame.display.set_mode(params.SCREEN_SIZE)
                self._rect = self._screen.get_rect()
                self._clock = pygame.time.Clock()
                self._init = True
            self._screen.fill(params.SIMULATION_BACKGROUND)
            self.display()
            pygame.display.flip()
            # frame = pygame.surfarray.array3d(self._screen)
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # frame = np.rot90(frame)  # Rotate because pygame's surface and OpenCV frame have different orientations
            # frame = cv2.flip(frame, 0)  # Flip vertically
            # self._video_writer.write(frame)

            self._clock.tick(fps)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

    def quit(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.unicode.isalpha():
                letter = event.unicode.upper()
                if letter == 'Q':
                    self._running = False
                    return True
        return False

    def save_data(self, data_name, path):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"An error occurred while creating the directory: {e}")
        path = f"{path+data_name}_{str(int(time.time()))}.pickle"
        with open(path, 'wb') as file:
            self._data_to_save["mean_dist_range"] = self._behaviors[0]._mean_dist_range
            pickle.dump(self._data_to_save, file)
