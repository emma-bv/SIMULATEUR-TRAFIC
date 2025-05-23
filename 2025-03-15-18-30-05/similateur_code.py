import traci

# Ajouter le chemin de SUMO à Python
#sumo_path = "C:/Program Files (x86)/Eclipse/Sumo/tools"
#sys.path.append(sumo_path)

# Chemin vers ton fichier de configuration SUMO
config_file = "osm.sumocfg"

# Démarrer SUMO avec TraCI
traci.start(["sumo-gui", "-c", config_file])  # Utilise "sumo" pour la version sans interface graphique

# Boucle de simulation
step = True
while step:
    traci.simulationStep()  # Avancer d'un pas de temps

    # Exemple : Récupérer la liste des véhicules
    vehicle_ids = traci.vehicle.getIDList()
    for veh_id in vehicle_ids:
        position = traci.vehicle.getPosition(veh_id)
        print(f"Véhicule {veh_id} est à la position {position}")

    # Récupérer la liste des feux de signalisation
    traffic_light_ids = traci.trafficlight.getIDList()
    print(f"Nombre de feux de signalisation : {len(traffic_light_ids)}")

    # Pour chaque feu de signalisation, récupérer la position et l'état
    for tl_id in traffic_light_ids:
        # Récupérer la position du feu de signalisation
        position = traci.junction.getPosition(tl_id)
        
        # Récupérer l'état actuel du feu de signalisation
        state = traci.trafficlight.getRedYellowGreenState(tl_id)
        
        # Afficher les informations
        print(f"Feu {tl_id} - Position : {position}, État : {state}")

    #step += 1

# Fermer TraCI
traci.close()