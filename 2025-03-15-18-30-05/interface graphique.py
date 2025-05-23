# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:39:13 2025

@author: user
"""

import sys
import traci
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QScrollArea, QGroupBox)
from PyQt5.QtCore import QTimer, Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from collections import deque

class SUMODashboard(QMainWindow):
    def __init__(self, config_file):
        super().__init__()
        self.config_file = config_file
        self.simulation_running = False
        self.data_history = {
            'vehicles': deque(maxlen=1000),
            'traffic_lights': deque(maxlen=1000),
            'positions': {}  # Pour stocker les positions historiques des véhicules
        }
        
        self.initUI()
        self.initSimulation()
        
    def initUI(self):
        self.setWindowTitle('SUMO Simulation Dashboard')
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout(central_widget)
        
        # Partie gauche - Graphiques
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Graphique des positions des véhicules
        self.position_fig = Figure(figsize=(10, 6), dpi=100)
        self.position_ax = self.position_fig.add_subplot(111)
        self.position_canvas = FigureCanvas(self.position_fig)
        left_layout.addWidget(self.position_canvas)
        
        # Graphique du nombre de véhicules
        self.vehicle_fig = Figure(figsize=(10, 3), dpi=100)
        self.vehicle_ax = self.vehicle_fig.add_subplot(111)
        self.vehicle_canvas = FigureCanvas(self.vehicle_fig)
        left_layout.addWidget(self.vehicle_canvas)
        
        # Graphique du nombre de feux
        self.trafficlight_fig = Figure(figsize=(10, 3), dpi=100)
        self.trafficlight_ax = self.trafficlight_fig.add_subplot(111)
        self.trafficlight_canvas = FigureCanvas(self.trafficlight_fig)
        left_layout.addWidget(self.trafficlight_canvas)
        
        main_layout.addWidget(left_widget, 70)  # 70% de l'espace
        
        # Partie droite - Informations et contrôles
        right_widget = QScrollArea()
        right_widget.setWidgetResizable(True)
        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        
        # Contrôles
        control_group = QGroupBox("Contrôles de Simulation")
        control_layout = QVBoxLayout()
        
        self.start_button = QPushButton("Démarrer Simulation")
        self.start_button.clicked.connect(self.toggle_simulation)
        control_layout.addWidget(self.start_button)
        
        self.step_button = QPushButton("Pas à Pas")
        self.step_button.clicked.connect(self.step_simulation)
        control_layout.addWidget(self.step_button)
        
        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)
        
        # Informations véhicules
        self.vehicle_group = QGroupBox("Informations Véhicules")
        self.vehicle_layout = QVBoxLayout()
        self.vehicle_group.setLayout(self.vehicle_layout)
        right_layout.addWidget(self.vehicle_group)
        
        # Informations feux
        self.trafficlight_group = QGroupBox("Informations Feux de Signalisation")
        self.trafficlight_layout = QVBoxLayout()
        self.trafficlight_group.setLayout(self.trafficlight_layout)
        right_layout.addWidget(self.trafficlight_group)
        
        right_widget.setWidget(right_content)
        main_layout.addWidget(right_widget, 30)  # 30% de l'espace
        
        # Timer pour la mise à jour des graphiques
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)  # Rafraîchissement toutes les 100ms
        
    def initSimulation(self):
        # Démarrer SUMO avec TraCI
        try:
            traci.start(["sumo-gui", "-c", self.config_file])
            self.simulation_running = True
            self.start_button.setText("Pause Simulation")
        except Exception as e:
            print(f"Erreur lors du démarrage de SUMO: {e}")
    
    def toggle_simulation(self):
        if self.simulation_running:
            self.simulation_running = False
            self.start_button.setText("Démarrer Simulation")
        else:
            self.simulation_running = True
            self.start_button.setText("Pause Simulation")
    
    def step_simulation(self):
        self.run_simulation_step()
    
    def run_simulation_step(self):
        if not self.simulation_running:
            return
            
        traci.simulationStep()
        
        # Récupérer les données des véhicules
        vehicle_ids = traci.vehicle.getIDList()
        vehicle_data = []
        
        # Effacer les anciennes informations
        for i in reversed(range(self.vehicle_layout.count())): 
            self.vehicle_layout.itemAt(i).widget().setParent(None)
            
        for veh_id in vehicle_ids:
            position = traci.vehicle.getPosition(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            vehicle_data.append((veh_id, position, speed))
            
            # Ajouter à l'historique des positions
            if veh_id not in self.data_history['positions']:
                self.data_history['positions'][veh_id] = deque(maxlen=100)
            self.data_history['positions'][veh_id].append(position)
            
            # Afficher les informations dans le panneau
            label = QLabel(f"{veh_id}: Pos={position}, Vitesse={speed:.2f}m/s")
            self.vehicle_layout.addWidget(label)
        
        self.data_history['vehicles'].append((traci.simulation.getTime(), len(vehicle_ids)))
        
        # Récupérer les données des feux
        traffic_light_ids = traci.trafficlight.getIDList()
        traffic_light_data = []
        
        # Effacer les anciennes informations
        for i in reversed(range(self.trafficlight_layout.count())): 
            self.trafficlight_layout.itemAt(i).widget().setParent(None)
            
        for tl_id in traffic_light_ids:
            position = traci.junction.getPosition(tl_id)
            state = traci.trafficlight.getRedYellowGreenState(tl_id)
            traffic_light_data.append((tl_id, position, state))
            
            # Afficher les informations dans le panneau
            label = QLabel(f"{tl_id}: Pos={position}, État={state}")
            self.trafficlight_layout.addWidget(label)
        
        self.data_history['traffic_lights'].append((traci.simulation.getTime(), len(traffic_light_ids)))
    
    def update_plots(self):
        if not self.simulation_running:
            return
            
        self.run_simulation_step()
        
        # Mettre à jour le graphique des positions
        self.position_ax.clear()
        
        # Afficher les positions actuelles des véhicules
        vehicle_ids = traci.vehicle.getIDList()
        for veh_id in vehicle_ids:
            position = traci.vehicle.getPosition(veh_id)
            self.position_ax.plot(position[0], position[1], 'bo')
            self.position_ax.text(position[0], position[1], veh_id, fontsize=8)
            
            # Afficher l'historique des positions
            if veh_id in self.data_history['positions']:
                positions = list(self.data_history['positions'][veh_id])
                x = [p[0] for p in positions]
                y = [p[1] for p in positions]
                self.position_ax.plot(x, y, 'b-', alpha=0.3)
        
        # Afficher les feux de signalisation
        traffic_light_ids = traci.trafficlight.getIDList()
        for tl_id in traffic_light_ids:
            position = traci.junction.getPosition(tl_id)
            self.position_ax.plot(position[0], position[1], 'rs', markersize=10)
            self.position_ax.text(position[0], position[1], tl_id, fontsize=8)
        
        self.position_ax.set_title('Positions des Véhicules et Feux')
        self.position_ax.set_xlabel('X (m)')
        self.position_ax.set_ylabel('Y (m)')
        self.position_ax.grid(True)
        self.position_canvas.draw()
        
        # Mettre à jour le graphique du nombre de véhicules
        self.vehicle_ax.clear()
        if self.data_history['vehicles']:
            times, counts = zip(*self.data_history['vehicles'])
            self.vehicle_ax.plot(times, counts, 'b-')
            self.vehicle_ax.set_title('Nombre de Véhicules au Cours du Temps')
            self.vehicle_ax.set_xlabel('Temps (s)')
            self.vehicle_ax.set_ylabel('Nombre de Véhicules')
            self.vehicle_ax.grid(True)
            self.vehicle_canvas.draw()
        
        # Mettre à jour le graphique du nombre de feux
        self.trafficlight_ax.clear()
        if self.data_history['traffic_lights']:
            times, counts = zip(*self.data_history['traffic_lights'])
            self.trafficlight_ax.plot(times, counts, 'r-')
            self.trafficlight_ax.set_title('Nombre de Feux de Signalisation Actifs')
            self.trafficlight_ax.set_xlabel('Temps (s)')
            self.trafficlight_ax.set_ylabel('Nombre de Feux')
            self.trafficlight_ax.grid(True)
            self.trafficlight_canvas.draw()
    
    def closeEvent(self, event):
        # Fermer proprement la simulation SUMO
        traci.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Chemin vers votre fichier de configuration SUMO
    config_file = "osm.sumocfg"
    
    dashboard = SUMODashboard(config_file)
    dashboard.show()
    
    sys.exit(app.exec_())