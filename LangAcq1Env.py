import sys
import json
import argparse
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box, Text, Discrete
import pygame
import random


class LangAcq1Env(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, config, render_mode: Optional[str] = None):
        self.stage_size = config["stage_size"]
        self.shapes = config["shapes"]
        self.colors = config["colors"]
        self.observation_space = Dict({"stage": Box(low=0, high=255,
                                                    shape=(self.stage_size, self.stage_size,
                                                           len(self.shapes) + len(self.colors)),
                                                    dtype=np.uint8),
                                       "text": Text(100)})
        self.grid_size = config["grid_size"]
        self.cardinality = config["cardinality"]
        self.objects = []
        self.render_wait = config["render_wait"]
        self.episode_length = config["episode_length"]
        self.image_path = config["image_path"]
        self.go_steps_limit = config["go_steps_limit"]
        self.push_steps_limit = config["push_steps_limit"]
        self.dictionary = config["dictionary"]
        self.images = {}
        for shape in self.shapes:
            self.images[shape] = {}
            for color in self.colors:
                self.images[shape][color] = self.fill_color(pygame.image.load(self.image_path + shape + ".jpg"), color)
        self.go_steps = np.zeros(self.cardinality, dtype=np.int8)
        self.push_steps = 0
        self.pusher = -1
        
        self.stage_image_size = self.stage_size * self.grid_size
        self.pos_xy = np.zeros((self.cardinality, 2), dtype=np.int8)
        self.dir_xy = np.zeros((self.cardinality, 2), dtype=np.int8)
        self.object_size = np.ones(self.cardinality, dtype=np.int8)
        self.brightness = np.ones(self.cardinality, dtype=np.int8)

        self.action_space = Discrete(2)
        self.stage = np.zeros((self.stage_size, self.stage_size, len(self.shapes) + len(self.colors)), dtype=np.uint8)
        self.text = ""
        self.render_mode = render_mode
        self.isopen = True
        self.episode_count = 0
        # self.dump = config['dump']

        pygame.init()
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.stage_image_size, self.stage_image_size))
        self.stage = pygame.Surface((self.stage_image_size, self.stage_image_size))
        self.clock = pygame.time.Clock()
        self.pygame_font = pygame.font.SysFont('Comic Sans MS', 14)

    def step(self, action):
        # calculate the next positions unhindered
        path = 0
        npuh = self.pos_xy + self.dir_xy
        pred = False
        # choose the subject
        subject = 1 if self.go_steps[0] == 0 and self.go_steps[1] > 0 else 0
        self.text = self.dictionary[self.objects[subject]["shape"]] + " "
        if np.random.randint(0, 10) < 3:
            self.text = self.text + self.dictionary[self.objects[subject]["color"]] + " "
        collided = False
        # wall collision
        for i in range(self.cardinality):
            if npuh[i][0] >= self.stage_size or npuh[i][0] < 0:   # the right/left wall
                self.dir_xy[i][0] = -1 * self.dir_xy[i][0]  # reverse direction
                npuh[i][0] = self.pos_xy[i][0]
                collided = True
                if not pred and subject == i:
                    self.text = self.text + "colpa le muro"
                    pred = True
                path = 1
            if npuh[i][1] >= self.stage_size or npuh[i][1] < 0:   # the upper/lower wall
                self.dir_xy[i][1] = -1 * self.dir_xy[i][1]  # reverse direction
                npuh[i][1] = self.pos_xy[i][1]
                collided = True
                if not pred and subject == i:
                    self.text = self.text + "colpa le muro"
                    pred = True
                path = 2
        if self.cardinality > 1:
            # object collision: if the next positions unhindered overlap
            if np.array_equal(npuh[0], npuh[1]) or \
                    (np.array_equal(self.pos_xy[0], npuh[1]) and np.array_equal(self.pos_xy[1], npuh[0])):  # flip
                collided = True
                if not pred:
                    self.text = self.text + "colpa " + self.dictionary[self.objects[1 - subject]["shape"]]
                    pred = True
                for j in range(2):  # x, y
                        if self.dir_xy[1][j] == 1 and self.pos_xy[0][j] >= self.stage_size - 2:
                            self.dir_xy[1][j] = -1
                        elif self.dir_xy[0][j] == 1 and self.pos_xy[1][j] >= self.stage_size - 2:
                            self.dir_xy[0][j] = -1
                        elif self.dir_xy[1][j] == -1 and self.pos_xy[0][j] <= 1:
                            self.dir_xy[1][j] = 1
                        elif self.dir_xy[0][j] == -1 and self.pos_xy[1][j] <= 1:
                            self.dir_xy[0][j] = 1
                        else:
                            if self.dir_xy[0][j] == 0:  # one's component j is 0; other is moving
                                self.dir_xy[0][j] = self.dir_xy[1][j]  # resuming other's motion
                                self.dir_xy[1][j] = 0
                                path = 4
                            elif self.dir_xy[1][j] == 0:  # one's component j is 0; other is moving
                                self.dir_xy[1][j] = self.dir_xy[0][j]  # resuming other's motion
                                self.dir_xy[0][j] = 0
                                path = 5
                            elif self.dir_xy[0][j] == -1 * self.dir_xy[0][j]:   # opposite direction
                                self.dir_xy[0][j] = -1 * self.dir_xy[0][j]  # reverse
                                self.dir_xy[1][j] = -1 * self.dir_xy[1][j]  # reverse
                                path = 6
                npuh = self.pos_xy + self.dir_xy
        self.pos_xy = npuh
        if not collided:
            for i in range(self.cardinality):
                if self.go_steps[i] > self.go_steps_limit:
                    self.dir_selection(i)
                    self.text = ""
                    self.go_steps[i] = 0
                else:
                    self.go_steps[i] += 1
                    if subject == i and not pred:
                        if self.go_steps[i] > 1:
                            pred = True
                            if np.abs(self.dir_xy[i]).max() == 0:
                                self.text = self.text + "pausa"
                            else:
                                distance = np.linalg.norm(self.pos_xy[i] - self.pos_xy[1 - i])
                                if abs(distance.max()) <= 1 and np.abs(self.dir_xy[1 - i]).max() == 0 and \
                                        0 < self.pos_xy[i, 0] < self.stage_size - 1 and \
                                        0 < self.pos_xy[i, 1] < self.stage_size - 1:
                                    self.text = self.text + "passa " + \
                                                self.dictionary[self.objects[1 - i]["shape"]]
                                else:
                                    self.text = self.text + "va"
                                    if self.dir_xy[i][1] == 1:
                                        self.text = self.text + " sub"
                                    elif self.dir_xy[i][1] == -1:
                                        self.text = self.text + " sup"
                                    if self.dir_xy[i][0] == 1:
                                        self.text = self.text + " dextre"
                                    elif self.dir_xy[i][0] == -1:
                                        self.text = self.text + " sinistre"
            # push
            if self.cardinality > 1:
                if self.push_steps > 0:
                    npuh = self.pos_xy + self.dir_xy
                    if npuh.min() < 0 or npuh.max() >= self.stage_size:
                        self.push_steps = 0
                    else:
                        self.pos_xy = npuh
                        self.push_steps += 1
                        self.text = self.dictionary[self.objects[self.pusher]["shape"]] + " pulsa " + \
                                    self.dictionary[self.objects[1 - self.pusher]["shape"]]
                        if np.random.randint(0, 10) < 3:
                            if self.dir_xy[self.pusher][1] == 1:
                                self.text = self.text + " sub"
                            elif self.dir_xy[self.pusher][1] == -1:
                                self.text = self.text + " sup"
                            if self.dir_xy[self.pusher][0] == 1:
                                self.text = self.text + " dextre"
                            elif self.dir_xy[self.pusher][0] == -1:
                                self.text = self.text + " sinistre"
                        pred = True
                    path = 7
                    if self.push_steps >= self.push_steps_limit:
                        self.push_steps = 0
                else:
                    for i in range(self.cardinality):
                        for j in range(self.cardinality):   # pusher
                            if i != j and np.random.randint(0, 2) == 1 and self.push_steps == 0:  # push mood start
                                self.pusher = j
                                if self.pos_xy[i][0] == self.pos_xy[j][0]:
                                    if self.pos_xy[i][1] == self.pos_xy[j][1] + 1 \
                                            and self.pos_xy[i][1] + self.push_steps_limit < self.stage_size:
                                        # j pushes i downward
                                        self.pos_xy[i][1] += 1
                                        self.pos_xy[j][1] += 1
                                        self.dir_xy[i][1] = 1
                                        self.dir_xy[j][1] = 1
                                        self.dir_xy[i][0] = 0
                                        self.dir_xy[j][0] = 0
                                        self.push_steps = 1
                                    if self.pos_xy[i][1] == self.pos_xy[j][1] - 1 \
                                            and self.pos_xy[i][1] - self.push_steps_limit > 0:
                                        # j pushes i upward
                                        self.pos_xy[i][1] += -1
                                        self.pos_xy[j][1] += -1
                                        self.dir_xy[i][1] = -1
                                        self.dir_xy[j][1] = -1
                                        self.dir_xy[i][0] = 0
                                        self.dir_xy[j][0] = 0
                                        self.push_steps = 1
                                if self.pos_xy[i][1] == self.pos_xy[j][1]:
                                    if self.pos_xy[i][0] == self.pos_xy[j][0] + 1 \
                                            and self.pos_xy[i][0] + self.push_steps_limit < self.stage_size:
                                        # j pushes i to the right
                                        self.pos_xy[i][0] += 1
                                        self.pos_xy[j][0] += 1
                                        self.dir_xy[i][0] = 1
                                        self.dir_xy[j][0] = 1
                                        self.dir_xy[i][1] = 0
                                        self.dir_xy[j][1] = 0
                                        self.push_steps = 1
                                    if self.pos_xy[i][0] == self.pos_xy[j][0] - 1 \
                                            and self.pos_xy[i][0] - self.push_steps_limit > 0:
                                        # j pushes i to the left
                                        self.pos_xy[i][0] += -1
                                        self.pos_xy[j][0] += -1
                                        self.dir_xy[i][0] = -1
                                        self.dir_xy[j][0] = -1
                                        self.dir_xy[i][1] = 0
                                        self.dir_xy[j][1] = 0
                                        self.push_steps = 1
        self.stage = np.zeros((self.stage_size, self.stage_size, len(self.shapes) + len(self.colors)), dtype=np.uint8)
        if not pred:
            self.text = ""
        observation = self._get_obs()
        self.episode_count += 1
        if self.episode_count >= self.episode_length:
            return observation, 0, True, False, {}
        else:
            return observation, 0, False, False, {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        observation = self._get_obs()
        text = observation["text"]
        self.screen = pygame.display.set_mode((self.stage_image_size, self.stage_image_size + 60))
        for i in range(self.cardinality):
            left = self.pos_xy[i, 0] * self.grid_size
            top = self.pos_xy[i, 1] * self.grid_size
            object = self.objects[i]
            image = self.images[object["shape"]][object["color"]]
            self.screen.blit(image, (left, top))
        text_surface = self.pygame_font.render(text, False, pygame.Color("white"))
        self.screen.blit(text_surface, (0, self.stage_image_size + 10))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()
        pygame.time.wait(self.render_wait)

    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        self.episode_count = 0
        self.objects = []
        self.go_steps = np.zeros(self.cardinality, dtype=np.int8)
        self.push_steps = 0
        self.pusher = -1
        random.shuffle(self.shapes)
        random.shuffle(self.colors)
        for i in range(self.cardinality):
            shape = self.shapes[i]
            color = self.colors[i]
            self.objects.append({"shape": shape, "color": color, "name": "O" + str(i+1)})
        pygame.init()
        self.stage = np.zeros((self.stage_size, self.stage_size, len(self.shapes) + len(self.colors)), dtype=np.uint8)
        self.text = ""
        observation = self._get_obs()
        pos_tmp = list(range(0, self.stage_size * self.stage_size))
        random.shuffle(pos_tmp)
        positions = pos_tmp[:2]
        self.dir_xy = np.zeros((self.cardinality, 2), dtype=np.int8)
        for i in range(self.cardinality):
            self.pos_xy[i, 0] = positions[i] % self.stage_size
            self.pos_xy[i, 1] = positions[i] // self.stage_size
            self.dir_selection(i)
        return observation, {}

    def dir_selection(self, i):            # centripetal direction selection (including 0)
        for j in range(2):  # x, y
            val = self.pos_xy[i][j] - 0.5 * (self.stage_size -
                                             np.random.randint(-1 * self.stage_size, self.stage_size + 1))
            if abs(val) < 0.25 * self.stage_size:
                self.dir_xy[i][j] = 0
            elif val > 0:
                self.dir_xy[i][j] = -1
            else:
                self.dir_xy[i][j] = 1

    def _get_obs(self):
        return {"stage": self.stage, "text": self.text}

    def close(self):
        pygame.display.quit()
        pygame.quit()
        self.isopen = False

    def fill_color(self, image, color):
        size = image.get_size()
        for i in range(size[0]):
            for j in range(size[1]):
                if image.get_at((i, j))[0] > 100:
                    image.set_at((i, j), pygame.Color(color))
                else:
                    image.set_at((i, j), pygame.Color("black"))
        return image


def run(env, episode_length, no_render, text_dump_f):
    env.reset()
    for i in range(episode_length):
        env.step(None)
        if text_dump_f is not None and env.text != "":
            text_dump_f.write(env.text + '\n')
        if not no_render:
            env.render()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file', default='LangAcq1Env.json')
    parser.add_argument('--epoch_length', help='epoch length', type=int, default=30, metavar='N')
    parser.add_argument('--epochs', help='epoch number', type=int, default=1, metavar='N')
    parser.add_argument('--text_dump', help='text dump file')
    parser.add_argument('--no_render', action='store_true')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    if config["cardinality"] > 2:
        print('Error: cardinality cannot be larger than 2!', file=sys.stderr)
        sys.exit(1)
    env = LangAcq1Env(config, render_mode="auto")
    print(isinstance(env.action_space, spaces.Box))
    if args.text_dump is not None:
        text_dump_f = open(args.text_dump, mode='w')
    else:
        text_dump_f = None
    for epoch in range(args.epochs):
        run(env, args.epoch_length, args.no_render, text_dump_f)
    env.close()
    if text_dump_f is not None:
        text_dump_f.close()


if __name__ == '__main__':
    main()
