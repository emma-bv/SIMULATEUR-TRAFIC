# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 22:54:53 2025

@author: user
"""


import numpy as np
import os
import sys

# Remplace ce chemin par le chemin d'installation de SUMO sur ton système
sumo_path = "C:/Program Files (x86)/Eclipse/Sumo/tools"
sys.path.append(sumo_path)

import traci
# Connexion à SUMO
sumo_binary = "sumo-gui"  # ou "sumo" si vous ne voulez pas l'interface graphique
sumo_config = "C:/Users/user/Sumo/2025-03-15-18-30-05/osm.sumocfg"
traci.start([sumo_binary, "-c", sumo_config])

# Définir les phases du feu de signalisation
PHASE_RED = 0
PHASE_GREEN = 1

# Définir les segments de route à surveiller
segments = ["segment1", "segment2", "segment3"]  # Remplacez par vos segments

# Fonction pour calculer la densité du trafic
def calculate_density(segment):
    vehicles = traci.edge.getLastStepVehicleIDs(segment)
    length = traci.edge.getLength(segment)
    return len(vehicles) / length

# Boucle de simulation
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    densities = {segment: calculate_density(segment) for segment in segments}
    max_density_segment = max(densities, key=densities.get)

    # Ajuster les feux de signalisation en fonction de la densité
    for segment in segments:
        if segment == max_density_segment:
            traci.trafficlight.setPhase("feu_id", PHASE_GREEN)  # Remplacez "feu_id" par l'ID de votre feu
        else:
            traci.trafficlight.setPhase("feu_id", PHASE_RED)

# Fermer la connexion à SUMO
traci.close()