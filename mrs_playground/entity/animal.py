# !/usr/bin/python3

import pygame
import numpy as np

from mrs_playground.common.entity import Entity


class Animal(Entity):
    def __init__(self,
                 pose: np.ndarray,
                 velocity: np.ndarray):
        super().__init__(
            pose=pose,
            velocity=velocity,
            image_path='red_triangle.png')

    def __str__(self) -> str:
        return "animal"

    def update(self, *args, **kwargs):
        pass

    def display(self, screen: pygame.Surface, debug=False):
        # Update graphics accordingly
        self._move(self._velocity)
        # pygame.draw.circle(screen, pygame.Color(
        #     "red2"), tuple(self._pose), 7)
        return super().display(screen, debug)
