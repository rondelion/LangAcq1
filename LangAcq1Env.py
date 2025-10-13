import sys
import json
import argparse
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Text
import pygame
import random
import copy


class LangAcq1Env(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, config, render_mode: Optional[str] = None):
        self.action_space = gym.spaces.Discrete(4)  # Dummy
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
        self.dictionary = config["dictionary"]
        self.vicinity_radius = config["vicinity_radius"]
        self.images = {}
        for shape in self.shapes:
            self.images[shape] = {}
            for color in self.colors:
                self.images[shape][color] = self.fill_color(pygame.image.load(self.image_path + shape + ".jpg"), color)
        self.go_steps = np.zeros(self.cardinality, dtype=np.int8)
        self.stage_image_size = self.stage_size * self.grid_size
        self.pos_xy = np.zeros((self.cardinality, 2), dtype=np.int8)
        self.dir_xy = np.zeros((self.cardinality, 2), dtype=np.int8)
        self.prev_dir = np.zeros((self.cardinality, 2), dtype=np.int8)  # for debug
        self.gira = [False, False]

        self.text = {}
        self.render_mode = render_mode
        self.episode_count = 0
        # self.dump = config['dump']

        pygame.init()
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.stage_image_size, self.stage_image_size))
        self.clock = pygame.time.Clock()
        self.pygame_font = pygame.font.SysFont('Comic Sans MS', 14)

    def step(self, action):
        # calculate the next positions unhindered
        path = [[0, 0], [0, 0]]
        prev_pos = copy.copy(self.pos_xy)
        prev_dir = copy.copy(self.dir_xy)
        self.prev_dir = copy.copy(self.dir_xy)  # for debug
        if self.cardinality > 1:
            prev_distance = np.linalg.norm(prev_pos[0] - prev_pos[1])
        npuh = self.pos_xy + self.dir_xy
        pred = {}
        # choose the subject
        # subject = 1 if self.cardinality > 1 and self.go_steps[0] == 0 and self.go_steps[1] > 0 else 0
        for i in range(self.cardinality):
            self.text[i] = self.dictionary[self.objects[i]["shape"]] + " "
            if np.random.randint(0, 10) < 3:
                self.text[i] = self.text[i] + self.dictionary[self.objects[i]["color"]] + " "
        collided = False
        # wall collision
        for i in range(self.cardinality):
            if npuh[i][0] >= self.stage_size or npuh[i][0] < 0:   # the right/left wall
                self.dir_xy[i][0] = -1 * self.dir_xy[i][0]  # reverse direction
                npuh[i][0] = self.pos_xy[i][0]
                collided = True
                self.go_steps[i] = 0
                if not pred.get(i, False):
                    self.text[i] = self.text[i] + "colpa le muro"
                    pred[i] = True
                # path[0] = 1  # x
            if npuh[i][1] >= self.stage_size or npuh[i][1] < 0:   # the upper/lower wall
                self.dir_xy[i][1] = -1 * self.dir_xy[i][1]  # reverse direction
                npuh[i][1] = self.pos_xy[i][1]
                collided = True
                self.go_steps[i] = 0
                if not pred.get(i, False):
                    self.text[i] = self.text[i] + "colpa le muro"
                    pred[i] = True
                # path[1] = 1  # y
        npuh_1 = copy.copy(npuh)
        if self.cardinality > 1:
            # object collision: if the next positions unhindered overlap
            if np.array_equal(npuh[0], npuh[1]) or \
                    (np.array_equal(self.pos_xy[0], npuh[1]) and np.array_equal(self.pos_xy[1], npuh[0])):  # flip
                collided = True
                for i in range(self.cardinality):
                    self.go_steps[i] = 0
                    if not pred.get(i, False):
                        self.text[i] = self.text[i] + "colpa " + self.dictionary[self.objects[1 - i]["shape"]]
                        pred[i] = True
                for j in range(2):  # x, y
                    if self.dir_xy[0][j] == 0:  # one's component j is 0; other is moving
                        self.dir_xy[0][j] = self.dir_xy[1][j]  # resuming other's motion
                        self.dir_xy[1][j] = 0
                        path[0][j] = 6
                    elif self.dir_xy[1][j] == 0:  # one's component j is 0; other is moving
                        self.dir_xy[1][j] = self.dir_xy[0][j]  # resuming other's motion
                        self.dir_xy[0][j] = 0
                        path[1][j] = 6
                    elif self.dir_xy[0][j] == -1 * self.dir_xy[1][j]:   # opposite direction
                        dir_0 = self.dir_xy[0][j]
                        self.dir_xy[0][j] = -1 * dir_0  # reverse
                        self.dir_xy[1][j] = dir_0  # reverse
                        path[0][j] = 8
                        path[1][j] = 8
                    if npuh[0][j] + self.dir_xy[0][j] < 0 or npuh[0][j] + self.dir_xy[0][j] >= self.stage_size:
                        self.dir_xy[0][j] = 0
                        path[0][j] = 9
                    if npuh[1][j] + self.dir_xy[1][j] < 0 or npuh[1][j] + self.dir_xy[1][j] >= self.stage_size:
                        self.dir_xy[1][j] = 0
                        path[1][j] = 9
                if np.array_equal((npuh + self.dir_xy)[0], (npuh + self.dir_xy)[1]):
                    npuh = self.pos_xy
                else:
                    npuh = npuh + self.dir_xy
            distance = np.linalg.norm(npuh[i] - npuh[1 - i])
        if not collided:
            for i in range(self.cardinality):
                if self.go_steps[i] > self.go_steps_limit:
                    self.dir_selection(i)
                    self.go_steps[i] = 0
                    if np.abs(prev_dir[i]).max() > 0 and np.abs(self.dir_xy[i]).max() > 0 and \
                            not np.array_equal(self.dir_xy[i], prev_dir[i]):
                        self.gira[i] = True
                        # print("t0", self.dir_xy[i], prev_dir[i])
                else:
                    self.go_steps[i] += 1
                if self.go_steps[i] > 1:
                    self.gira[i] = False
                if not pred.get(i, False):
                    pred[i] = True
                    if np.abs(self.dir_xy[i]).max() == 0:
                        self.text[i] = self.text[i] + "pausa"
                    elif self.gira[i] and self.go_steps[i] > 0:
                        self.text[i] = self.text[i] + "gira"
                        # print("t1", self.dir_xy[i], prev_dir[i])
                    elif self.cardinality > 1 and self.go_steps[i] > 1 and \
                            np.abs(npuh[i][0] - npuh[1 - i][0]) <= self.vicinity_radius and \
                            np.abs(npuh[i][1] - npuh[1 - i][1]) <= self.vicinity_radius and \
                            distance >= prev_distance and \
                            not np.array_equal(self.dir_xy[i], self.dir_xy[1 - i]):
                        self.text[i] = self.text[i] + "passa " + self.dictionary[self.objects[1 - i]["shape"]]
                    else:
                        self.text[i] = self.text[i] + "va"
                        if self.dir_xy[i][1] == 1:
                            self.text[i] = self.text[i] + " sub"
                        elif self.dir_xy[i][1] == -1:
                            self.text[i] = self.text[i] + " sup"
                        if self.dir_xy[i][0] == 1:
                            self.text[i] = self.text[i] + " dextre"
                        elif self.dir_xy[i][0] == -1:
                            self.text[i] = self.text[i] + " sinistre"
                        if self.cardinality > 1 and \
                                np.abs(npuh[i][0] - npuh[1 - i][0]) <= self.vicinity_radius and \
                                np.abs(npuh[i][1] - npuh[1 - i][1]) <= self.vicinity_radius and \
                                np.array_equal(self.dir_xy[i], self.dir_xy[1-i]):
                            self.text[i] = self.text[i] + " con " + self.dictionary[self.objects[1 - i]["shape"]]
        self.pos_xy = npuh
        for i in range(self.cardinality):
            if not pred.get(i, False):
                self.text[i] = ""
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
        text = observation["text"][0]
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
        shapes = random.sample(self.shapes, len(self.shapes))
        colors = random.sample(self.colors, len(self.colors))
        for i in range(self.cardinality):
            shape = shapes[i]
            color = colors[i]
            self.objects.append({"shape": shape, "color": color, "name": "O" + str(i+1)})
        pygame.init()
        self.text = {}
        pos_tmp = list(range(0, self.stage_size * self.stage_size))
        random.shuffle(pos_tmp)
        positions = pos_tmp[:2]
        self.dir_xy = np.zeros((self.cardinality, 2), dtype=np.int8)
        for i in range(self.cardinality):
            self.pos_xy[i, 0] = positions[i] % self.stage_size
            self.pos_xy[i, 1] = positions[i] // self.stage_size
            self.dir_selection(i)
        return self._get_obs(), {}

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
        stage = np.zeros((self.stage_size, self.stage_size, len(self.shapes) + len(self.colors)), dtype=np.uint8)
        for i in range(self.cardinality):
            shape = np.zeros(len(self.shapes), dtype=np.uint8)
            shape[self.shapes.index(self.objects[i]['shape'])] = 1
            color = np.zeros(len(self.colors), dtype=np.uint8)
            color[self.colors.index(self.objects[i]['color'])] = 1
            stage[self.pos_xy[i][0], self.pos_xy[i][1]] = np.concatenate([shape, color], 0)
        return {"stage": stage, "text": self.text}

    def close(self):
        pygame.display.quit()
        pygame.quit()

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
        if text_dump_f is not None:
            for j in range(env.cardinality):
                if env.text[j] != "":
                    text_dump_f.write(env.text[j] + '\n')
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
