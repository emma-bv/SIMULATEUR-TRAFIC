# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 22:30:20 2025

@author: user
"""

import traci

# Chemin vers ton fichier de configuration SUMO
config_file = "osm.sumocfg"

# Démarrer SUMO avec TraCI
traci.start(["sumo-gui", "-c", config_file])  # Utilise "sumo" pour la version sans interface graphique

# Boucle de simulation
step = True
while step:
    traci.simulationStep()  # Avancer d'un pas de temps

    # Récupérer la liste des feux de signalisation
    traffic_light_ids = traci.trafficlight.getIDList()

    # Pour chaque feu de signalisation, récupérer la position, l'état et la densité de véhicules
    for tl_id in traffic_light_ids:
        # Récupérer la position du feu de signalisation
        position = traci.junction.getPosition(tl_id)

        # Récupérer l'état actuel du feu de signalisation
        state = traci.trafficlight.getRedYellowGreenState(tl_id)

        # Récupérer les identifiants des voies connectées au feu de signalisation
        lane_ids = traci.trafficlight.getControlledLanes(tl_id)

        # Calculer la densité de véhicules sur chaque voie connectée
        lane_densities = {}
        for lane_id in lane_ids:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            lane_length = traci.lane.getLength(lane_id)
            if lane_length > 0:
                density = len(vehicle_ids) / lane_length
            else:
                density = 0
            lane_densities[lane_id] = density

        # Afficher les informations
        print(f"Feu {tl_id} - Position : {position}, État : {state}")
        for lane_id, density in lane_densities.items():
            print(f"  Voie {lane_id} - Densité : {density:.2f} véhicules/mètre")

    # step += 1

# Fermer TraCI
traci.close()