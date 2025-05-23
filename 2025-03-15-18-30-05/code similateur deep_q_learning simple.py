# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 23:47:23 2025

@author: user
"""

import traci
import numpy as np
import random
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Configuration de SUMO
config_file = "osm.sumocfg"
traci.start(["sumo-gui", "-c", config_file])

# Paramètres DQL
STATE_SIZE = 4  # Par exemple, densité des voies autour du feu
ACTION_SIZE = 3  # Nombre d'actions possibles (ex: changer le feu, rester, alterner)
GAMMA = 0.95
ALPHA = 0.001
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 2000
BATCH_SIZE = 32

# Durée initiale du feu
BASE_GREEN_DURATION = 10  # Durée de base du feu vert
MAX_GREEN_DURATION = 30  # Durée maximale possible

def build_model():
    model = Sequential([
        Dense(24, input_dim=STATE_SIZE, activation='relu'),
        Dense(24, activation='relu'),
        Dense(ACTION_SIZE, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=ALPHA))
    return model

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = build_model()
        self.epsilon = EPSILON

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += GAMMA * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

agent = DQNAgent()

def get_state():
    traffic_light_ids = traci.trafficlight.getIDList()
    state = []
    queue_lengths = {}  # Dictionnaire pour stocker les files d'attente
    
    for tl_id in traffic_light_ids:
        densities = [len(traci.lane.getLastStepVehicleIDs(lane)) for lane in traci.trafficlight.getControlledLanes(tl_id)]
        queue_lengths[tl_id] = sum(densities)  # Total de véhicules en attente
        state.extend(densities)
    
    # Ajuster la taille de l'état pour correspondre à STATE_SIZE
    if len(state) > STATE_SIZE:
        state = state[:STATE_SIZE]  # Tronquer
    else:
        state += [0] * (STATE_SIZE - len(state))  # Compléter avec des zéros
    
    return np.reshape(np.array(state), [1, STATE_SIZE]), queue_lengths

for episode in range(1000):  # Nombre d'épisodes d'apprentissage
    state, queue_lengths = get_state()
    done = False
    
    while not done:
        action = agent.act(state)
        traci.simulationStep()
        next_state, queue_lengths = get_state()
        reward = -sum(state[0])  # Récompense négative si congestion
        done = False  # Définir une condition d'arrêt
        
        # Déterminer le feu de signalisation avec la plus longue file d'attente
        max_queue_tl = max(queue_lengths, key=queue_lengths.get)
        max_queue_length = queue_lengths[max_queue_tl]
        
        # Ajuster la durée du feu
        green_duration = min(BASE_GREEN_DURATION + max_queue_length // 2, MAX_GREEN_DURATION)
        traci.trafficlight.setPhaseDuration(max_queue_tl, green_duration)
        
        print(f"Épisode {episode}: Priorité à {max_queue_tl} avec {max_queue_length} véhicules en attente. Durée ajustée à {green_duration}s")
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
    
    agent.replay()

traci.close()
