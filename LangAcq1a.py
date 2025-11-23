import sys
import json
import os
import re
import argparse
import numpy as np
import gymnasium as gym
import pandas as pd

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import cerenaut_pt_core.utils as cerenaut_utils
import LangAcq1L


class LangAcq1a:
    def __init__(self, config, env, glm):
        self.stage_size = config['env']["stage_size"]
        self.env = env
        self.glm = glm
        self.observation, _ = self.env.reset()
        self.no_render = config['env']['no_render']
        self.stage_buffer = [[{} for i in range(self.stage_size)] for i in range(self.stage_size)]
        self.saliency_map = np.zeros((self.stage_size, self.stage_size))
        # self.fm_map = np.zeros((self.stage_size, self.stage_size))  # feature_matching_map
        self.gaze_pos = np.array([-1, -1])
        self.color = np.zeros(len(config['env']['colors']), dtype=np.int8)
        self.shape = np.zeros(len(config['env']['shapes']), dtype=np.int8)
        self.vicinity_radius = config['env']['vicinity_radius']
        self.motion = np.array([0, 0])
        self.border = np.zeros((2, 2), dtype=np.int8)
        self.vicinity = np.zeros((3, 3), dtype=np.int8)
        self.turn = 0
        self.after_saccade = 0
        self.mode = config['mode']
        self.prev_text = None  # for debug
        self.prev_prev_text = None  # for debug
        self.prev_prev_prev_text = None  # for debug
        self.prev_motion = {}  # for debug
        self.prev_prev_motion = {}  # for debug

    def step(self):
        prev_stage = self.observation['stage']  # for debug
        self.observation, _, _, _, _ = self.env.step(None)
        if not self.no_render:
            self.env.render()
        stage = self.observation['stage']
        text = self.observation['text']
        self.make_stage_buffer(stage, text[0])
        self.make_saliency_map(stage, prev_stage)
        if text[0] != '':
            if self.mode == 'learn':
                if self.glm.text_filter is None or re.search(self.glm.text_filter, text[0]):
                    self.glm.text_learn(text[0], stage, self)
            elif self.mode == 'eval':
                text_decoded = self.glm.text_predict(stage, self)
                self.glm.eval_predict(text, text_decoded, stage, self)
        self.prev_prev_prev_text = self.prev_prev_text.copy() if self.prev_prev_text is not None else None
        self.prev_prev_text = self.prev_text.copy() if self.prev_text is not None else None
        self.prev_text = text.copy()

    def make_stage_buffer(self, stage, text):  # text for debug
        stage_buffer = [[{} for i in range(self.stage_size)] for i in range(self.stage_size)]
        for i in range(self.stage_size):
            for j in range(self.stage_size):
                stage_buffer[i][j]['shape'] = stage[i, j][:4]  # shape
                stage_buffer[i][j]['color'] = stage[i, j][4:]  # color
                stage_buffer[i][j]['motion'] = np.zeros(2)
                prev_border = self.stage_buffer[i][j].get('border')  # for debug
                turn = 0
                if max(stage_buffer[i][j]['shape']) > 0 and max(stage_buffer[i][j]['color']) > 0 \
                        and 'shape' in self.stage_buffer[0][0]:
                    prev_motion = np.zeros(2)
                    for k in range(self.stage_size):
                        for l in range(self.stage_size):
                            if np.array_equal(self.stage_buffer[k][l]['shape'], stage_buffer[i][j]['shape']) and \
                                    np.array_equal(self.stage_buffer[k][l]['color'], stage_buffer[i][j]['color']):
                                stage_buffer[i][j]['prev_pos'] = np.array([k, l])
                                motion = np.array(
                                    list(map(lambda x: 1 if x > 0 else -1 if x < 0 else 0,
                                             np.array([i - k, j - l]))))  # normalize to -1, 0, 1
                                stage_buffer[i][j]['motion'] = motion
                                prev_motion = self.stage_buffer[k][l]['motion'].copy()
                                stage_buffer[i][j]['prev_motion'] = prev_motion.copy()
                                prev_prev_motion = self.stage_buffer[k][l].get('prev_motion', np.zeros(2)).copy()
                                self.prev_motion[tuple(np.array(self.stage_buffer[k][l]['shape']))] = prev_motion.copy()
                                self.prev_prev_motion[tuple(np.array(self.stage_buffer[k][l]['shape']))] = \
                                    prev_prev_motion.copy()
                                if max(abs(motion)) > 0 and not np.array_equal(motion, prev_motion) and \
                                        max(abs(prev_motion)) > 0:  # or
                                    turn = 1
                                else:
                                    turn = 0
                stage_buffer[i][j]['turn'] = turn
                border = np.zeros((2, 2), dtype=np.int8)
                if i == 0:
                    border[0, 0] = 1
                elif i == self.stage_size - 1:
                    border[0, 1] = 1
                if j == 0:
                    border[1, 0] = 1
                elif j == self.stage_size - 1:
                    border[1, 1] = 1
                stage_buffer[i][j]['border'] = border
        vicinity_diameter = (self.vicinity_radius * 2) + 1
        for i in range(self.stage_size):
            for j in range(self.stage_size):
                vicinity = [0, 0, 0, 0, 0]
                if np.max(stage[i, j]) > 0:
                    for k in range(vicinity_diameter):
                        for l in range(vicinity_diameter):
                            x = i + k - self.vicinity_radius
                            y = j + l - self.vicinity_radius
                            if 0 <= x < self.stage_size and 0 <= y < self.stage_size:
                                if np.max(stage[x, y]) > 0 and not (x == i and y == j):
                                    predicted_pos = np.array([[0, 0], [0, 0]])
                                    if 'prev_pos' in stage_buffer[i][j] and 'prev_motion' in stage_buffer[i][j]:
                                        predicted_pos[0] = stage_buffer[i][j]['prev_pos'] \
                                                           + stage_buffer[i][j]['prev_motion']
                                    else:
                                        predicted_pos[0] = np.array([i, j])
                                    if 'prev_pos' in stage_buffer[x][y] and 'prev_motion' in stage_buffer[x][y]:
                                        predicted_pos[1] = stage_buffer[x][y]['prev_pos'] \
                                                           + stage_buffer[x][y]['prev_motion']
                                    else:
                                        predicted_pos[1] = np.array([x, y])
                                    collision = 1 if np.array_equal(predicted_pos[0], predicted_pos[1]) else 0
                                    go_along = 0
                                    not_approaching = 0
                                    approaching = 0
                                    if np.max(np.abs(stage_buffer[i][j]['motion'])) > 0:
                                        if np.array_equal(stage_buffer[i][j]['motion'],
                                                          stage_buffer[x][y]['motion']):
                                            go_along = 1
                                        else:
                                            prev_distance = np.linalg.norm(stage_buffer[i][j]['prev_pos'] -
                                                                           stage_buffer[x][y]['prev_pos'])
                                            distance = np.linalg.norm(np.array([i, j]) - np.array([x, y]))
                                            if distance >= prev_distance:
                                                not_approaching = 1
                                            else:
                                                approaching = 1
                                    vicinity = [1, collision, go_along, not_approaching, approaching]
                stage_buffer[i][j]['vicinity'] = np.array(vicinity)
        self.stage_buffer = stage_buffer.copy()

    def make_saliency_map(self, stage, prev_stage):  # prev_stage for debug
        self.saliency_map = np.zeros((self.stage_size, self.stage_size))
        for i in range(self.stage_size):
            for j in range(self.stage_size):
                if stage[i][j].max() > 0:
                    self.saliency_map[i][j] = 1

    def get_feature_matching_map(self, stage, shape, color):
        fm_map = np.zeros((self.stage_size, self.stage_size))
        for i in range(self.stage_size):
            for j in range(self.stage_size):
                fm_map[i][j] = self.saliency_map[i][j] + (stage[i][j][:4] @ shape + stage[i][j][4:] @ color) * 2
        return fm_map

    def saccade(self, stage, pre_shape, pre_color, forced):
        saccade = False
        fm_map = self.get_feature_matching_map(stage, pre_shape, pre_color)
        if forced:
            fm_map[self.gaze_pos[0], self.gaze_pos[1]] = \
                fm_map[self.gaze_pos[0], self.gaze_pos[1]] * 0.1  # suppress the first (prev.) max
            pos = np.array(np.unravel_index(np.argmax(fm_map), fm_map.shape))  # second max
        elif np.max(pre_shape) > 0 or np.max(pre_color) > 0 or np.array_equal([-1, -1], self.gaze_pos):
            pos = np.array(np.unravel_index(np.argmax(fm_map), fm_map.shape))  # first max
        else:
            pos = self.gaze_pos
        if not np.array_equal(pos, self.gaze_pos) and not np.array_equal([-1, -1], self.gaze_pos):
            saccade = True
        self.gaze_pos = pos
        self.shape = stage[pos[0]][pos[1]][:4]
        self.color = stage[pos[0]][pos[1]][4:]
        if 'vicinity' in self.stage_buffer[pos[0]][pos[1]]:
            self.vicinity = self.stage_buffer[pos[0]][pos[1]]['vicinity']
        else:
            self.vicinity = np.zeros(4)
        self.motion = self.stage_buffer[pos[0]][pos[1]]['motion']
        self.border = self.stage_buffer[pos[0]][pos[1]]['border']
        self.turn = self.stage_buffer[pos[0]][pos[1]]['turn']  # self.stage_buffer[pos[0]][pos[1]]['prev_turn'])
        if saccade:
            self.after_saccade = 1
        return saccade

    def find_shape(self, stage, shape):  # for evaluation
        pos = [-1, -1]
        for i in range(self.stage_size):
            for j in range(self.stage_size):
                if stage[i][j][:4] @ shape > 0:
                    pos = [i, j]
        return pos

    def glm_train(self, epoch):
        self.glm.batch_train(epoch)

    def reset(self):
        self.observation, _ = self.env.reset()
        self.stage_buffer = [[{} for i in range(self.stage_size)] for i in range(self.stage_size)]
        self.prev_motion = {}
        self.prev_prev_motion = {}


class GroundedLM:  # grounded language model
    def __init__(self, config, embeddings_file, vocabulary, config_env):
        self.config_env = config_env  # for debug
        self.vocabulary = vocabulary
        self.idx2word = self.make_idx2word()
        use_cuda = not config["no_cuda"] and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        config["device"] = self.device
        self.v_size = len(vocabulary)
        self.embeddings = self.load_embeddings(embeddings_file, self.v_size)
        self.iteration = config["train_iteration"]
        self.log_interval = config['log_interval']
        self.max_text_len = config['max_text_len']
        self.input_dim = self.embeddings.shape[1] + 4 + 4 + 5 + 5 + 2 + 2 + 1
        # shape, color, vicinity, motion1h, border, turn, after_saccade
        if config["word_predictor"]["type"] == "cerenaut_pt_simple_ae":
            from cerenaut_pt_core.components.simple_autoencoder import SimpleAutoencoder
            self.word_predictor = SimpleAutoencoder((-1, self.input_dim),
                                                    config["word_predictor"],
                                                    (-1, self.v_size))
            self.wp_optimizer = eval("optim." + config['word_predictor']['optimizer'] +
                                     "(self.word_predictor.parameters(), lr=config['word_predictor']['learning_rate'])")
            self.activation_func = eval(config['word_predictor']["activation_func"])
            if "activation_func_dim" in config['word_predictor']:
                self.activation_func.dim = config['word_predictor']["activation_func_dim"]
            self.wp_loss_func = eval(config["word_predictor"]["loss_func"])
            self.wp_type = "cerenaut_pt_simple_ae"
        elif config["word_predictor"]["type"] == "simple":
            from SimplePredictor import Perceptron
            self.activation_func = eval(config['word_predictor']["activation_func"])
            if "activation_func_dim" in config['word_predictor']:
                self.activation_func.dim = config['word_predictor']["activation_func_dim"]
            self.word_predictor = Perceptron(self.input_dim,
                                             config['word_predictor']['hidden_dim'],
                                             self.v_size,
                                             None)
            self.wp_optimizer = eval("optim." + config['word_predictor']['optimizer'] +
                                     "(self.word_predictor.parameters(), lr=config['word_predictor']['learning_rate'])")
            self.wp_loss_func = eval(config["word_predictor"]["loss_func"])
            self.wp_type = "simple"
        elif config["word_predictor"]["type"] == "linear":
            self.activation_func = eval(config['word_predictor']["activation_func"])
            if "activation_func_dim" in config['word_predictor']:
                self.activation_func.dim = config['word_predictor']["activation_func_dim"]
            self.word_predictor = nn.Linear(self.input_dim, self.v_size)
            self.wp_optimizer = eval("optim." + config['word_predictor']['optimizer'] +
                                     "(self.word_predictor.parameters(), lr=config['word_predictor']['learning_rate'])")
            self.wp_loss_func = eval(config["word_predictor"]["loss_func"])
            self.wp_type = "linear"
        else:
            print("Word Predictor Model type " + "'" + config["word_predictor"]['type'] + "' is not supported!")
            exit(-1)
        if config["saccade_predictor"]["type"] == "cerenaut_pt_simple_ae":
            from cerenaut_pt_core.components.simple_autoencoder import SimpleAutoencoder
            embedding_size = self.embeddings.shape[1]
            self.saccade_predictor = SimpleAutoencoder((-1, embedding_size * 2), config["saccade_predictor"], (-1, 2))
            self.sp_optimizer = eval("optim." + config['saccade_predictor']['optimizer'] +
                                     "(self.saccade_predictor.parameters(), lr=config['saccade_predictor']['learning_rate'])")
            self.sp_loss_func = eval(config["saccade_predictor"]["loss_func"])
        else:
            print("Saccade Predictor Model type " + "'" + config["saccade_predictor"]['type'] + "' is not supported!")
            exit(-1)
        self.embedding_classifier = None
        self.word2color = np.zeros((len(self.vocabulary), len(config_env['colors'])))
        self.word2shape = np.zeros((len(self.vocabulary), len(config_env['shapes'])))
        self.word2feature_modified = False
        self.word_predictor_dataset = GroundedLM.GLMDataset()
        self.saccade_predictor_dataset = GroundedLM.GLMDataset()
        self.text_filter = config["text_filter"] if config["text_filter"] != "" else None
        self.glm_stat_dump = config.get('glm_stat_dump', None)
        self.diff_output = config.get('diff_output', None)
        self.saccade_cnt = 0
        self.saccade_predicted_cnt = 0
        self.saccade_correctly_predicted_cnt = 0
        self.dictionary_R = {}
        for k, v in config_env['dictionary'].items():
            self.dictionary_R[v] = k
        self.eval = {"total": 0,
                     "excess": 0,
                     "subject": {"prd": {}, "mtc": {}},
                     "color": {"env": {}, "prd": {}, "mtc": {}},
                     "verb": {"env": {}, "prd": {}, "mtc": {}},
                     "direction": {"env": 0, "prd": 0, "mtc": 0},
                     "con": {"env": 0, "prd": 0, "mtc": 0},
                     "object": {"env": {}, "prd": {}, "mtc": {}},
                     "colpa_object": {"env": 0, "prd": 0, "mtc": 0}}

    class GLMDataset(Dataset):
        def __init__(self):
            self.array = []

        def __getitem__(self, index):
            return torch.tensor(self.array[index][0], dtype=torch.float32), \
                   torch.tensor(self.array[index][1], dtype=torch.float32)

        def __len__(self):
            return len(self.array)

        def add(self, input_vector, output):
            self.array.append([output, input_vector])

        def clear(self):
            self.array = []

    @staticmethod
    def get_token_feature(token_vector, word2feature):
        token_feature = token_vector @ word2feature
        token_feature = token_feature / token_feature.sum() if token_feature.sum() > 0 else token_feature
        return token_feature if token_feature.max() > 0.9 else np.zeros(token_feature.size)

    def learn_word2features(self, token_vector, color, shape):
        if self.word2color.max() < 10000:
            self.word2color = self.word2color + np.dot(token_vector.reshape((-1, 1)), color.reshape((1, color.size)))
            self.word2feature_modified = True
        if self.word2shape.max() < 10000:
            self.word2shape = self.word2shape + np.dot(token_vector.reshape((-1, 1)), shape.reshape((1, shape.size)))
            self.word2feature_modified = True

    @staticmethod
    def get_motion1h(motion):
        motion1h = np.zeros(5)  # one hot vector
        if motion[0] < 0:
            motion1h[0] = 1
        elif motion[0] > 0:
            motion1h[1] = 1
        if motion[1] < 0:
            motion1h[2] = 1
        elif motion[1] > 0:
            motion1h[3] = 1
        if np.max(np.abs(motion)) == 0:  # no motion
            motion1h[4] = 1
        return motion1h

    @staticmethod
    def binary1hot(x):
        return np.array([1 - x, x])

    def text_learn(self, text, stage, agent):
        agent.gaze_pos = np.array([-1, -1])
        agent.after_saccade = 0
        if len(text) == 0:
            return
        tokens = LangAcq1L.Lexicon.tokenizer(text, self.vocabulary)
        word = 'EOS'
        for token in tokens:
            if token == 0:  # BOS > EOS
                token = 1
                bos = True
            else:
                bos = False
            embedding = torch.from_numpy(self.embeddings[token].astype(np.float32)).clone()
            if not bos:
                token_vector = np.zeros(self.v_size)
                token_vector[token] = 1
                word = self.idx2word[token]  # for debug
                token_color = self.get_token_feature(token_vector, self.word2color)
                token_shape = self.get_token_feature(token_vector, self.word2shape)
                # to the feature-matched object
                saccade = agent.saccade(stage, token_shape, token_color, False)
                predicted_saccade = self.predict_saccade(prev_embedding, embedding)
                if saccade:
                    self.saccade_cnt += 1
                if predicted_saccade:
                    self.saccade_predicted_cnt += 1
                if saccade and predicted_saccade:
                    self.saccade_correctly_predicted_cnt += 1
                self.add_word_predictor_dataset(agent, prev_embedding, token_vector)
                self.add_saccade_predictor_dataset(prev_embedding, embedding, saccade)
                self.stat_dump(prev_word, agent, word)
                self.learn_word2features(token_vector, agent.color, agent.shape)
            prev_embedding = embedding
            prev_word = word  # for debug

    def get_context(self, agent):
        motion1h = self.get_motion1h(agent.motion)
        border = self.binary1hot(agent.border.max())
        turn = self.binary1hot(agent.turn)
        return torch.from_numpy(np.concatenate((agent.shape, agent.color, agent.vicinity, motion1h,
                                                border, turn, np.array([agent.after_saccade]))).astype(
            np.float32)).clone()

    def add_word_predictor_dataset(self, agent, prev_embedding, token_vector):
        context = self.get_context(agent)
        input_vector = torch.cat((prev_embedding, context)).float().to(self.device)
        self.word_predictor_dataset.add(input_vector, token_vector)

    def stat_dump(self, prev_word, agent, word):
        if self.glm_stat_dump is not None:
            data = [prev_word, word, agent.border.max(), agent.turn]
            data.extend(agent.motion)
            data.extend(agent.vicinity)
            self.glm_stat_dump.write('\t'.join(map(str, data)))
            self.glm_stat_dump.write('\n')

    def add_saccade_predictor_dataset(self, prev_embedding, embedding, go):
        input_vector = torch.cat((prev_embedding, embedding)).float().to(self.device)
        go_nogo = torch.from_numpy((np.array([0, 1]) if go else np.array([1, 0])).astype(np.float32)).clone()
        self.saccade_predictor_dataset.add(input_vector, go_nogo)

    def text_predict(self, stage, agent):
        agent.gaze_pos = np.array([-1, -1])
        agent.after_saccade = 0
        text_decoded = ""
        word_decoded = ""
        no_shape = np.zeros(self.word2shape.shape[1], dtype=np.int8)
        no_color = np.zeros(self.word2color.shape[1], dtype=np.int8)
        cnt = 0
        prev_embedding = torch.from_numpy(self.embeddings[self.vocabulary['EOS']].astype(np.float32)).clone()
        while cnt < self.max_text_len and word_decoded != "EOS":
            agent.saccade(stage, no_shape, no_color, False)  # first take
            token_decoded = self.predict_word(agent, prev_embedding)
            embedding = torch.from_numpy(self.embeddings[token_decoded].astype(np.float32)).clone()
            forced_saccade = self.predict_saccade(prev_embedding, embedding)
            if forced_saccade:
                agent.saccade(stage, no_shape, no_color, True)  # retake features
                token_decoded = self.predict_word(agent, prev_embedding)
            embedding = torch.from_numpy(self.embeddings[token_decoded].astype(np.float32)).clone()
            word_decoded = self.idx2word[token_decoded]
            text_decoded = word_decoded if text_decoded == "" else text_decoded + " " + word_decoded
            prev_embedding = embedding
            cnt += 1
        return text_decoded

    def predict_word(self, agent, prev_embedding):
        self.word_predictor.eval()
        context = self.get_context(agent)
        input_vector = torch.from_numpy(np.zeros((1, self.input_dim)).astype(np.float32)).clone()
        input_vector[0] = torch.cat((prev_embedding, context)).float().to(self.device)
        if self.wp_type == "cerenaut_pt_simple_ae":
            _, decoded = self.word_predictor(input_vector)
            decoded = self.activation_func(decoded)
        else:
            decoded = self.activation_func(self.word_predictor(input_vector))
        decoded_token_vector = np.asarray(decoded[0].to('cpu').detach().numpy().copy()).astype('float64')
        multinominal = np.random.multinomial(1, decoded_token_vector / decoded_token_vector.sum())
        return np.argmax(multinominal)

    def predict_saccade(self, prev_embedding, embedding):
        self.saccade_predictor.eval()
        input_vector = torch.from_numpy(np.zeros((1, torch.numel(embedding) * 2)).astype(np.float32)).clone()
        input_vector[0] = torch.cat((prev_embedding, embedding)).float().to(self.device)
        _, predicted_go_nogo = self.saccade_predictor(input_vector)
        return (torch.argmax(predicted_go_nogo) == 1).item()

    def word_predictor_train(self, epoch):
        dataloader = DataLoader(self.word_predictor_dataset, batch_size=10, shuffle=True)
        self.word_predictor.train()
        train_loss = 0.0
        for i in range(self.iteration):
            for batch_idx, data in enumerate(dataloader):
                token_vector = data[0]
                input_vector = data[1]
                self.wp_optimizer.zero_grad()
                if self.wp_type == "cerenaut_pt_simple_ae":
                    _, decoded = self.word_predictor(input_vector)
                else:
                    decoded = self.word_predictor(input_vector)
                loss = self.wp_loss_func(decoded, token_vector)
                loss.backward()
                self.wp_optimizer.step()
                train_loss += loss.item()
        train_loss = train_loss / (self.iteration * batch_idx) if batch_idx > 0 else 0.0
        print('GroundedLM WP Epoch: {}\tLoss: {:.6f}'.format(epoch, train_loss))

    def saccade_predictor_train(self, epoch):
        dataloader = DataLoader(self.saccade_predictor_dataset, batch_size=10, shuffle=True)
        self.saccade_predictor.train()
        train_loss = 0.0
        for i in range(self.iteration):
            for batch_idx, data in enumerate(dataloader):
                go_nogo_vector = data[0]
                input_vector = data[1]
                self.sp_optimizer.zero_grad()
                _, decoded = self.saccade_predictor(input_vector)
                loss = self.sp_loss_func(decoded, go_nogo_vector)
                loss.backward()
                self.sp_optimizer.step()
                train_loss += loss.item()
        train_loss = train_loss / (self.iteration * batch_idx) if batch_idx > 0 else 0.0
        print('GroundedLM SP Epoch: {}\tLoss: {:.6f}'.format(epoch, train_loss))

    def batch_train(self, epoch):
        if self.word_predictor_dataset.__len__() > 0:
            self.word_predictor_train(epoch)
        if self.saccade_predictor_dataset.__len__() > 0:
            self.saccade_predictor_train(epoch)
        self.word_predictor_dataset.clear()
        self.saccade_predictor_dataset.clear()

    def load_embeddings(self, file, v_size):
        buf = pd.read_csv(file)
        buf = np.array(buf.values.tolist())
        shape = buf.shape
        embeddings = np.zeros((v_size, shape[1] - 1))
        for line in buf:
            word = line[-1]
            idx = self.vocabulary[word]
            embeddings[idx] = line[:-1]
        return embeddings

    def make_idx2word(self):
        idx2word = {}
        for key in self.vocabulary:
            idx2word[self.vocabulary[key]] = key
        return idx2word

    def dump_variables_in_train(self, shape, color):
        shape_str = self.config_env['shapes'][np.argmax(shape)]
        color_str = self.config_env['colors'][np.argmax(color)]
        return shape_str, color_str

    def eval_predict(self, text, text_decoded, stage, agent):
        shape = ""
        text_d_buf = text_decoded.strip().split()
        if text_d_buf[-1] == 'EOS':
            text_d_buf.pop()
        self.eval["total"] += 1
        if len(text_d_buf) > 0:
            text_buf = {0: text[0].split(), 1: text[1].split()}  # texts from the environment
            subject = text_d_buf[0]
            if subject in self.dictionary_R and self.dictionary_R[subject] in self.config_env['shapes']:
                shape = self.dictionary_R[subject]
                self.eval["subject"]["prd"][subject] = self.eval["subject"]["prd"].get(subject, 0) + 1
                if len(text_buf[0]) > 0 and subject == text_buf[0][0]:
                    self.eval["subject"]["mtc"][subject] = self.eval["subject"]["mtc"].get(subject, 0) + 1
                    text_idx = 1
                elif len(text_buf[1]) > 0 and subject == text_buf[1][0]:
                    self.eval["subject"]["mtc"][subject] = self.eval["subject"]["mtc"].get(subject, 0) + 1
                    text_idx = 2
                elif len(shape) > 0 and (len(text_buf[0]) == 0 or len(text_buf[1]) == 0):
                    text_idx = -1  # one of the counter texts is missing
                else:
                    text_idx = -2  # subject not in the scene
            else:
                text_idx = -2  # subject not in the scene
            if text_idx > 0:
                shape_vector = np.zeros(len(self.config_env['shapes']), dtype=np.int8)
                shape_vector[self.config_env['shapes'].index(shape)] = 1
                pos = agent.find_shape(stage, shape_vector)
                if len(text_d_buf) > 1 and text_d_buf[1] in self.dictionary_R and \
                        self.dictionary_R[text_d_buf[1]] in self.config_env['colors']:
                    next_word_d_index = 2
                    color_vector = stage[pos[0]][pos[1]][4:]
                    color_idx = np.argmax(color_vector)
                    color = self.config_env['colors'][color_idx]
                    self.eval["color"]["prd"][color] = self.eval["color"]["prd"].get(color, 0) + 1
                    if color == self.dictionary_R[text_d_buf[1]]:
                        self.eval["color"]["mtc"][color] = self.eval["color"]["mtc"].get(color, 0) + 1
                else:
                    next_word_d_index = 1
                color_in_text = self.dictionary_R.get(text_buf[text_idx - 1][1], None)
                if color_in_text in self.config_env['colors']:
                    next_word_index = 2
                    self.eval["color"]["env"][color_in_text] = self.eval["color"]["env"].get(color_in_text, 0) + 1
                else:
                    next_word_index = 1
                verb = text_buf[text_idx - 1][next_word_index]
                self.eval["verb"]["env"][verb] = self.eval["verb"]["env"].get(verb, 0) + 1
                if len(text_d_buf) > next_word_d_index and text_d_buf[next_word_d_index] in self.config_env['verbs']:
                    verb_d = text_d_buf[next_word_d_index]
                    self.eval["verb"]["prd"][verb_d] = self.eval["verb"]["prd"].get(verb_d, 0) + 1
                    if verb == verb_d:
                        self.eval["verb"]["mtc"][verb] = self.eval["verb"]["mtc"].get(verb, 0) + 1
                        next_word_index += 1
                        next_word_d_index += 1
                        if verb == 'pausa' or verb == 'gira':
                            if len(text_d_buf) > next_word_d_index:
                                self.eval["excess"] += 1
                        elif verb == 'va':
                            lim_d = text_d_buf.index("con") if "con" in text_d_buf else len(text_d_buf)
                            dirs_d = set(text_d_buf[next_word_d_index:lim_d]) if len(text_d_buf) > next_word_d_index \
                                else set([])
                            if len(dirs_d) > 0 and text_d_buf[next_word_d_index] \
                                    in ["sup", "sub", "dextre", "sinistre"]:
                                self.eval["direction"]["prd"] += 1
                            lim = text_buf[text_idx - 1].index("con") if "con" in text_buf[text_idx - 1] \
                                else len(text_buf[text_idx - 1])
                            dirs = set(text_buf[text_idx - 1][next_word_index:lim])
                            if text_buf[text_idx - 1][next_word_index] in ["sup", "sub", "dextre", "sinistre"]:
                                self.eval["direction"]["env"] += 1
                            if dirs_d == dirs:
                                self.eval["direction"]["mtc"] += 1
                            if "con" in text_buf[text_idx - 1]:
                                self.eval["con"]["env"] += 1
                                if len(text_d_buf) > lim_d and text_d_buf[lim_d] == "con":
                                    self.eval["con"]["prd"] += 1
                                    self.eval["con"]["mtc"] += 1
                                    if len(text_d_buf) > lim_d + 1:
                                        self.eval_predict_object_check(text_buf[text_idx - 1][lim + 1],
                                                                       text_d_buf[lim_d + 1])
                                    if len(text_d_buf) > lim_d + 2:
                                        self.eval["excess"] += 1
                            elif "con" in text_d_buf:
                                self.eval["con"]["prd"] += 1
                        elif verb == "passa":
                            self.eval_predict_object_check(text_buf[text_idx - 1][next_word_index],
                                                           text_d_buf[next_word_d_index])
                            if len(text_d_buf) > next_word_d_index + 1:
                                self.eval["excess"] += 1
                        elif verb == "colpa":
                            obj = text_buf[text_idx - 1][next_word_index]
                            if obj in self.dictionary_R and self.dictionary_R[obj] in self.config_env['shapes']:
                                self.eval["colpa_object"]["env"] += 1
                            if len(text_d_buf) > next_word_d_index:
                                obj_d = text_d_buf[next_word_d_index]
                                if obj_d in self.dictionary_R and self.dictionary_R[obj_d] in self.config_env['shapes']:
                                    self.eval["colpa_object"]["prd"] += 1
                                    if obj_d == obj:
                                        self.eval["colpa_object"]["mtc"] += 1
                                self.eval_predict_object_check(obj, obj_d)
                                if "le muro" in text_decoded:
                                    if len(text_d_buf) > next_word_d_index + 2:
                                        self.eval["excess"] += 1
                                else:
                                    if len(text_d_buf) > next_word_d_index + 1:
                                        self.eval["excess"] += 1
                if self.diff_output is not None:
                    self.diff_output.write('> {}\n'.format(text[text_idx - 1]))
                    self.diff_output.write('< {}\n'.format(text_decoded))

    def eval_predict_object_check(self, obj, obj_d):
        if obj in self.dictionary_R and self.dictionary_R[obj] in self.config_env['shapes']:
            self.eval["object"]["env"][obj] = self.eval["object"]["env"].get(obj, 0) + 1
        if obj_d in self.dictionary_R and self.dictionary_R[obj_d] in self.config_env['shapes']:
            self.eval["object"]["prd"][obj_d] = self.eval["object"]["prd"].get(obj_d, 0) + 1
            if obj_d == obj:
                self.eval["object"]["mtc"][obj] = self.eval["object"]["mtc"].get(obj, 0) + 1

    def make_eval_predict_table(self, dic, header, values, title):
        if "Trifolio" in dic or "green" in dic or "va" in dic:
            keys = sorted(dic)
            for key in keys:
                values = values + str(dic[key]) + "\t"
                header = header + title + ":" + key + "\t"
        else:
            for key in dic:
                title1 = key if title == "" else title + ':' + key
                if type(dic[key]) is dict:
                    header, values = self.make_eval_predict_table(dic[key], header, values, title1)
                else:
                    header = header + title1
                    values = values + str(dic[key])
                if header[-1] != "\t":
                    header = header + "\t"
                if values[-1] != "\t":
                    values = values + "\t"
        return header, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config file', default='LangAcq1.json')
    parser.add_argument('--epoch_length', help='epoch length', type=int, default=30, metavar='N')
    parser.add_argument('--epochs', help='epoch number', type=int, default=1, metavar='N')
    parser.add_argument('--no_render', action='store_true')
    parser.add_argument('--word2features', help='word to features model file', default='word2features.npz')
    parser.add_argument('--vocabulary', help='vocabulary file', default='vocabulary.json')
    parser.add_argument('--embeddings', help='embeddings csv file', default='features.csv')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='sentence count for loss output')
    parser.add_argument('--wp_dump', help='word predictor model dump file path', default='word_predictor.pt')
    parser.add_argument('--sp_dump', help='saccade predictor model dump file path', default='saccade_predictor.pt')
    parser.add_argument('--glm_stat_dump', help='grounded lang model train data dump file path')
    parser.add_argument('--mode', help='learn or eval', choices=['learn', 'eval'])
    parser.add_argument('--diff_output', help='predicted text and corresponding text file')
    parser.add_argument('--eval_json', help='evaluation json file')
    parser.add_argument('--eval_tsv', help='evaluation tsv file')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    if config["env"]["cardinality"] > 2:
        print('Error: cardinality cannot be larger than 2!', file=sys.stderr)
        sys.exit(1)
    env = gym.make("LangAcq1Env-v0", config=config["env"], render_mode="human")
    with open(args.vocabulary) as vocabulary_file:
        vocabulary = json.load(vocabulary_file)
    config["groundedLM"]['log_interval'] = args.log_interval
    if args.glm_stat_dump is not None:
        config["groundedLM"]['glm_stat_dump'] = open(args.glm_stat_dump, mode='w')
    if args.mode == 'eval' and args.diff_output is not None:
        config["groundedLM"]['diff_output'] = open(args.diff_output, mode='w')
    glm = GroundedLM(config["groundedLM"], args.embeddings, vocabulary, config["env"])
    if os.path.isfile(args.wp_dump):
        print("Loading word predictor model: " + args.wp_dump)
        checkpoint = torch.load(args.wp_dump, weights_only=True)
        if 'word_predictor' in checkpoint:
            glm.word_predictor.load_state_dict(checkpoint['word_predictor'])
        else:
            glm.word_predictor.load_state_dict(torch.load(args.wp_dump, weights_only=True))
    if os.path.isfile(args.sp_dump):
        print("Loading saccade predictor model: " + args.sp_dump)
        checkpoint = torch.load(args.sp_dump, weights_only=True)
        if 'saccade_predictor' in checkpoint:
            glm.saccade_predictor.load_state_dict(checkpoint['saccade_predictor'])
        else:
            glm.saccade_predictor.load_state_dict(torch.load(args.sp_dump, weights_only=True))
    if os.path.isfile(args.word2features):
        print("Loading word to features model: " + args.word2features)
        word2features = np.load(args.word2features)
        glm.word2shape = word2features['arr_0']
        glm.word2color = word2features['arr_1']
    config['env']['no_render'] = args.no_render
    config['mode'] = args.mode
    agent = LangAcq1a(config, env, glm)
    for epoch in range(args.epochs):
        for i in range(args.epoch_length):
            agent.step()
        if args.mode == 'learn':
            agent.glm_train(epoch)
        agent.reset()
    env.close()
    if args.glm_stat_dump is not None:
        config["groundedLM"]['glm_stat_dump'].close()
    if args.mode == 'learn':
        torch.save(glm.word_predictor.state_dict(), args.wp_dump)
        torch.save(glm.saccade_predictor.state_dict(), args.sp_dump)
        if glm.word2feature_modified:
            np.savez(args.word2features, glm.word2shape, glm.word2color)
    else:
        if args.diff_output is not None:
            config["groundedLM"]['diff_output'].close()
        if args.eval_json is not None:
            with open(args.eval_json, 'w') as f:
                json.dump(glm.eval, f, indent=2)
        if args.eval_tsv is not None:
            header = ""
            values = ""
            title = ""
            header, values = glm.make_eval_predict_table(glm.eval, header, values, title)
            with open(args.eval_tsv, 'w') as f:
                f.write(header + '\n')
                f.write(values + '\n')


if __name__ == '__main__':
    main()
