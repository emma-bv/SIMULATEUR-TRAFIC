# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 18:36:43 2025

@author: user
"""

import traci
import random
import numpy as np

# Paramètres de simulation
config_file = "osm.sumocfg"
simulation_steps = 100000

# Paramètres Q-learning
alpha = 0.1  # Taux d'apprentissage
gamma = 0.9  # Facteur de réduction
epsilon = 0.1  # Taux d'exploration

# Initialisation de la table Q
q_table = {}

def get_state(tl_id):
    """Récupère l'état du feu de signalisation (nombre de véhicules en attente)"""
    state = 0
    for lane in traci.trafficlight.getControlledLanes(tl_id):
        state += traci.lane.getLastStepHaltingNumber(lane)
    return state

def get_reward(tl_id):
    """Calcule la récompense (négative du nombre de véhicules en attente)"""
    reward = 0
    for lane in traci.trafficlight.getControlledLanes(tl_id):
        reward -= traci.lane.getLastStepHaltingNumber(lane)
    return reward

def choose_action(tl_id, state):
    """Choisit une action (changer l'état du feu) en utilisant epsilon-greedy"""
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1])  # Exploration
    else:
        if (tl_id, state) not in q_table:
            return random.choice([0, 1])
        else:
            return np.argmax(q_table[(tl_id, state)])  # Exploitation

def apply_action(tl_id, action):
    """Applique l'action (changer l'état du feu)"""
    current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
    if action == 1:
        # Changer l'état (exemple simple : inversion des feux)
        new_state = ""
        for s in current_state:
            if s == "r":
                new_state += "g"
            elif s == "g":
                new_state += "r"
            else:
                new_state += s
        traci.trafficlight.setRedYellowGreenState(tl_id, new_state)

def update_q_table(tl_id, state, action, reward, next_state):
    """Met à jour la table Q"""
    if (tl_id, state) not in q_table:
        q_table[(tl_id, state)] = [0, 0]
    if (tl_id, next_state) not in q_table:
        q_table[(tl_id, next_state)] = [0, 0]

    q_table[(tl_id, state)][action] = q_table[(tl_id, state)][action] + alpha * (reward + gamma * np.max(q_table[(tl_id, next_state)]) - q_table[(tl_id, state)][action])

# Démarrer SUMO avec TraCI
traci.start(["sumo-gui", "-c", config_file])

# Boucle de simulation
for step in range(simulation_steps):
    traci.simulationStep()

    # Contrôle des feux de signalisation avec Q-learning
    for tl_id in traci.trafficlight.getIDList():
        state = get_state(tl_id)
        action = choose_action(tl_id, state)
        apply_action(tl_id, action)
        next_state = get_state(tl_id)
        reward = get_reward(tl_id)
        update_q_table(tl_id, state, action, reward, next_state)

# Fermer TraCI
traci.close()