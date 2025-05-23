import traci
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from collections import deque

# Paramètres du RL
alpha = 0.1  # Taux d'apprentissage
gamma = 0.9  # Facteur de récompense
epsilon = 0.1  # Probabilité d'exploration
memory = deque(maxlen=2000)  # Mémoire pour l'expérience replay
batch_size = 32

def build_model():
    """Construire le réseau neuronal pour DQN"""
    model = keras.Sequential([
        keras.layers.Dense(24, input_dim=1, activation='relu'),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(4, activation='linear')  # 4 actions (durée du feu)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    return model

model = build_model()

# Historique des métriques
rewards_history = []
avg_speed_history = []
vehicle_density_history = []

def get_state(tl_id):
    """Récupère l'état actuel du trafic autour du feu de signalisation"""
    vehicle_ids = traci.vehicle.getIDList()
    congestion_level = sum(traci.vehicle.getSpeed(veh) < 2 for veh in vehicle_ids)  # Nombre de véhicules à l'arrêt
    return np.array([congestion_level])


def choose_action(state):
    """Sélectionne une action via le modèle DQN"""
    if np.random.rand() < epsilon:
        return random.choice([5, 10, 15, 20])  # Exploration
    q_values = model.predict(state, verbose=0)
    return [5, 10, 15, 20][np.argmax(q_values)]


def remember(state, action, reward, new_state):
    """Stocker l'expérience pour entraînement futur"""
    memory.append((state, action, reward, new_state))


def replay():
    """Réentraîner le modèle avec l'expérience replay"""
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, new_state in minibatch:
        target = reward + gamma * np.max(model.predict(new_state, verbose=0))
        target_f = model.predict(state, verbose=0)
        target_f[0][[5, 10, 15, 20].index(action)] = target
        model.fit(state, target_f, epochs=1, verbose=0)

# Démarrer SUMO
config_file = "osm.sumocfg"
traci.start(["sumo-gui", "-c", config_file])

# Boucle de simulation
for step in range(1000):
    traci.simulationStep()
    total_reward = 0
    total_speed = 0
    total_vehicles = 0
    
    traffic_light_ids = traci.trafficlight.getIDList()
    for idx, tl_id in enumerate(traffic_light_ids):
        tl_name = f"feu_{idx}"
        state = get_state(tl_id)
        
        if state[0] == 0:
            continue
        
        action = choose_action(state)
        traci.trafficlight.setPhaseDuration(tl_id, action)
        
        new_state = get_state(tl_id)
        reward = -new_state[0]
        remember(state, action, reward, new_state)
        total_reward += reward
        
        print(f"{tl_name} - État actuel: {traci.trafficlight.getRedYellowGreenState(tl_id)}")
        
        segment_vehicles = traci.lanearea.getIDList()
        for segment in segment_vehicles:
            num_vehicles = traci.lanearea.getLastStepVehicleNumber(segment)
            avg_speed = traci.lanearea.getLastStepMeanSpeed(segment)
            total_speed += avg_speed
            total_vehicles += num_vehicles
            print(f"Segment {segment}: {num_vehicles} véhicules, Vitesse Moyenne: {avg_speed:.2f} m/s")
    
    rewards_history.append(total_reward)
    avg_speed_history.append(total_speed / (total_vehicles + 1e-5))
    vehicle_density_history.append(total_vehicles)
    replay()

traci.close()

# Affichage des métriques sous forme de graphiques
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(rewards_history, label='Récompense cumulée')
plt.xlabel('Étapes')
plt.ylabel('Récompense')
plt.title('Évolution de la récompense')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(avg_speed_history, label='Vitesse Moyenne')
plt.xlabel('Étapes')
plt.ylabel('Vitesse (m/s)')
plt.title('Évolution de la vitesse moyenne')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(vehicle_density_history, label='Densité de véhicules')
plt.xlabel('Étapes')
plt.ylabel('Nombre de véhicules')
plt.title('Évolution de la densité de trafic')
plt.legend()

plt.tight_layout()
plt.show()
