import traci
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Configuration de SUMO
sumo_binary = "sumo-gui"  # ou "sumo" pour la version sans interface graphique
sumo_config = "osm.sumocfg"

# Environnement personnalisé pour SUMO
class SumoEnv(gym.Env):
    def __init__(self):
        super(SumoEnv, self).__init__()
        # Définir l'espace d'action et d'observation
        self.action_space = gym.spaces.Discrete(2)  # Actions : 0 = rouge, 1 = vert
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)  # Exemple d'état

        # Démarrer SUMO
        traci.start([sumo_binary, "-c", sumo_config])

        # Vérifier les arêtes disponibles
        self.edge_ids = traci.edge.getIDList()
        if not self.edge_ids:
            raise ValueError("Aucune arête trouvée dans le réseau SUMO.")
        print("Arêtes disponibles :", self.edge_ids)

        # Utiliser la première arête disponible
        self.edge_id = self.edge_ids[0]

    def reset(self, seed=None, options=None):
        # Recharger la simulation
        traci.load(["-c", sumo_config])
        return self._get_state(), {}  # Retourner l'état et un dictionnaire vide (info)

    def step(self, action):
        # Appliquer l'action (changer le feu)
        if action == 0:
            traci.trafficlight.setRedYellowGreenState("tl1", "rrrrGG")  # Feu rouge
        else:
            traci.trafficlight.setRedYellowGreenState("tl1", "GGrrrr")  # Feu vert

        # Avancer d'un pas de temps
        traci.simulationStep()

        # Obtenir l'état suivant
        next_state = self._get_state()

        # Calculer la récompense
        reward = self._calculate_reward()

        # Vérifier si la simulation est terminée
        done = traci.simulation.getMinExpectedNumber() == 0

        # Info supplémentaire (optionnel)
        info = {}

        return next_state, reward, done, False, info  # Retourner next_state, reward, terminated, truncated, info

    def _get_state(self):
        # Exemple d'état : Densité du trafic, longueur de la file d'attente, temps d'attente
        vehicles = traci.edge.getLastStepVehicleIDs(self.edge_id)
        density = len(vehicles)
        queue_length = traci.edge.getLastStepHaltingNumber(self.edge_id)
        waiting_time = traci.edge.getWaitingTime(self.edge_id)
        return np.array([density, queue_length, waiting_time], dtype=np.float32)

    def _calculate_reward(self):
        # Récompense basée sur la réduction de la file d'attente et du temps d'attente
        queue_length = traci.edge.getLastStepHaltingNumber(self.edge_id)
        waiting_time = traci.edge.getWaitingTime(self.edge_id)
        reward = - (queue_length + waiting_time)  # Récompense négative pour minimiser la congestion
        return reward

    def close(self):
        traci.close()

# Créer l'environnement
env = SumoEnv()

# Vérifier que l'environnement est correctement défini
check_env(env)

# Créer le modèle PPO
model = PPO("MlpPolicy", env, verbose=1)

# Entraîner le modèle
model.learn(total_timesteps=10000)

# Tester le modèle
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    print(f"Action : {action}, Récompense : {rewards}")

# Fermer l'environnement
env.close()