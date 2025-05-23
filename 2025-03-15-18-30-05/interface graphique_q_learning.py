# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 17:03:59 2025

@author: user
"""

import sys
import traci
import random
import numpy as np
from collections import defaultdict, deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QPushButton, QGroupBox, QScrollArea,
                             QComboBox, QSlider, QCheckBox)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class SimulationThread(QThread):
    update_signal = pyqtSignal()

    def __init__(self, config_file):
        super().__init__()
        self.config_file = config_file
        self.running = False
        self.paused = False
        self.speed = 1
        self.q_table = defaultdict(lambda: [0, 0])
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def run(self):
        traci.start(["sumo-gui", "-c", self.config_file, "--start", "--quit-on-end"])
        self.running = True

        while self.running:
            if not self.paused:
                for _ in range(self.speed):
                    traci.simulationStep()
                    self.run_qlearning_step()

            self.update_signal.emit()

            QThread.msleep(50)  # Réduire la charge CPU

    def run_qlearning_step(self):
        for tl_id in traci.trafficlight.getIDList():
            state = self.get_state(tl_id)
            action = self.choose_action(tl_id, state)
            self.apply_action(tl_id, action)
            next_state = self.get_state(tl_id)
            reward = self.get_reward(tl_id)
            self.update_q_table(tl_id, state, action, reward, next_state)

    def get_state(self, tl_id):
        return min(sum(traci.lane.getLastStepHaltingNumber(lane)
                        for lane in traci.trafficlight.getControlledLanes(tl_id)), 10)

    def get_reward(self, tl_id):
        return -sum(traci.lane.getLastStepHaltingNumber(lane)
                    for lane in traci.trafficlight.getControlledLanes(tl_id))

    def choose_action(self, tl_id, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1])
        return np.argmax(self.q_table[(tl_id, state)])

    def apply_action(self, tl_id, action):
        if action == 1:
            current = traci.trafficlight.getRedYellowGreenState(tl_id)
            traci.trafficlight.setRedYellowGreenState(tl_id,
                                                     ''.join({'r':'g', 'g':'r'}.get(c, c) for c in current))

    def update_q_table(self, tl_id, state, action, reward, next_state):
        current_q = self.q_table[(tl_id, state)][action]
        max_next_q = np.max(self.q_table[(tl_id, next_state)])
        self.q_table[(tl_id, state)][action] = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q)

    def stop(self):
        self.running = False
        self.wait()
        traci.close()

class TrafficDashboard(QMainWindow):
    def __init__(self, config_file):
        super().__init__()
        self.config_file = config_file

        # Configuration initiale
        self.update_interval = 200  # ms
        self.show_trajectories = True
        self.show_vehicle_ids = True

        # Données pour visualisation
        self.vehicle_history = defaultdict(lambda: deque(maxlen=10))
        self.congestion_data = deque(maxlen=100)
        self.action_data = deque(maxlen=100)

        self.initUI()
        self.initThread()

    def initUI(self):
        self.setWindowTitle('Contrôle Optimal des Feux - Dashboard Fluide')
        self.setGeometry(100, 100, 1000, 700)

        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Partie gauche - Visualisation (70%)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Carte des positions
        self.map_fig = Figure(figsize=(7, 5), dpi=90)
        self.map_ax = self.map_fig.add_subplot(111)
        self.map_canvas = FigureCanvas(self.map_fig)
        left_layout.addWidget(self.map_canvas, 60)

        # Graphiques de performance
        self.metrics_fig = Figure(figsize=(7, 3), dpi=90)
        self.congestion_ax = self.metrics_fig.add_subplot(121)
        self.actions_ax = self.metrics_fig.add_subplot(122)
        self.metrics_canvas = FigureCanvas(self.metrics_fig)
        left_layout.addWidget(self.metrics_canvas, 40)

        main_layout.addWidget(left_widget, 70)

        # Partie droite - Contrôles (30%)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Contrôles de simulation
        control_group = QGroupBox("Contrôle Simulation")
        control_layout = QVBoxLayout()

        self.start_btn = QPushButton("Démarrer")
        self.start_btn.clicked.connect(self.toggle_simulation)
        control_layout.addWidget(self.start_btn)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(1)
        self.speed_slider.valueChanged.connect(self.update_speed)
        control_layout.addWidget(QLabel("Vitesse:"))
        control_layout.addWidget(self.speed_slider)

        self.trajectory_cb = QCheckBox("Afficher trajectoires")
        self.trajectory_cb.setChecked(True)
        self.trajectory_cb.stateChanged.connect(self.toggle_trajectories)
        control_layout.addWidget(self.trajectory_cb)

        self.ids_cb = QCheckBox("Afficher IDs")
        self.ids_cb.setChecked(True)
        self.ids_cb.stateChanged.connect(self.toggle_ids)
        control_layout.addWidget(self.ids_cb)

        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)

        # Informations Q-learning
        q_group = QGroupBox("Apprentissage par Renforcement")
        q_layout = QVBoxLayout()

        self.tl_combo = QComboBox()
        q_layout.addWidget(QLabel("Feu sélectionné:"))
        q_layout.addWidget(self.tl_combo)

        self.q_label = QLabel("Sélectionnez un feu pour voir sa table Q")
        self.q_label.setWordWrap(True)
        q_layout.addWidget(self.q_label)

        q_group.setLayout(q_layout)
        right_layout.addWidget(q_group)

        # Informations temps réel
        info_group = QGroupBox("Statistiques Temps Réel")
        self.info_layout = QVBoxLayout()
        info_group.setLayout(self.info_layout)
        right_layout.addWidget(info_group)

        right_layout.addStretch()
        right_scroll.setWidget(right_widget)
        main_layout.addWidget(right_scroll, 30)

        # Timer pour mise à jour UI
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(self.update_interval)

    def initThread(self):
        self.sim_thread = SimulationThread(self.config_file)
        self.sim_thread.update_signal.connect(self.update_data)

    def toggle_simulation(self):
        if not self.sim_thread.isRunning():
            self.sim_thread.start()
            self.start_btn.setText("Pause")
        else:
            self.sim_thread.paused = not self.sim_thread.paused
            self.start_btn.setText("Reprendre" if self.sim_thread.paused else "Pause")

    def update_speed(self, value):
        self.sim_thread.speed = value

    def toggle_trajectories(self, state):
        self.show_trajectories = state == Qt.Checked
        self.update_map()

    def toggle_ids(self, state):
        self.show_vehicle_ids = state == Qt.Checked
        self.update_map()

    def update_data(self):
        # Mettre à jour les données de visualisation
        for veh_id in traci.vehicle.getIDList():
            self.vehicle_history[veh_id].append(traci.vehicle.getPosition(veh_id))

        # Calculer la congestion totale
        congestion = sum(traci.lane.getLastStepHaltingNumber(lane)
                         for tl in traci.trafficlight.getIDList()
                         for lane in traci.trafficlight.getControlledLanes(tl))
        self.congestion_data.append(congestion)

        # Mettre à jour la liste des feux
        current_tl = self.tl_combo.currentText()
        self.tl_combo.clear()
        self.tl_combo.addItems(traci.trafficlight.getIDList())
        if current_tl in traci.trafficlight.getIDList():
            self.tl_combo.setCurrentText(current_tl)

    def update_ui(self):
        self.update_map()
        self.update_metrics()
        self.update_info()
        self.update_q_info()

    def update_map(self):
        self.map_ax.clear()

        # Afficher les véhicules
        for veh_id, positions in self.vehicle_history.items():
            if positions:
                x, y = zip(*positions)
                if self.show_trajectories and len(x) > 1:
                    self.map_ax.plot(x, y, 'b-', alpha=0.3, linewidth=0.5)
                self.map_ax.plot(x[-1], y[-1], 'bo', markersize=4)
                if self.show_vehicle_ids:
                    self.map_ax.text(x[-1], y[-1], veh_id, fontsize=6)

        # Afficher les feux
        for tl_id in traci.trafficlight.getIDList():
            pos = traci.junction.getPosition(tl_id)
            state = traci.trafficlight.getRedYellowGreenState(tl_id)
            color = 'red' if 'r' in state else 'green'
            self.map_ax.plot(pos[0], pos[1], 's', color=color, markersize=8)
            self.map_ax.text(pos[0], pos[1], tl_id, fontsize=7)

        self.map_ax.set_title('Carte du Trafic en Temps Réel')
        self.map_ax.grid(True)
        self.map_canvas.draw()

    def update_metrics(self):
        self.congestion_ax.clear()
        self.actions_ax.clear()

        # Graphique de congestion
        if self.congestion_data:
            self.congestion_ax.plot(self.congestion_data, 'r-')
        self.congestion_ax.set_title('Congestion Totale')
        self.congestion_ax.grid(True)

        # Graphique d'actions (simplifié)
        if hasattr(self.sim_thread, 'action_count'):
            self.actions_ax.bar(['Maintien', 'Changement'],
                                 [self.sim_thread.action_count.get(0, 0),
                                  self.sim_thread.action_count.get(1, 0)])
        self.actions_ax.set_title('Actions RL')
        self.actions_ax.grid(True)

        self.metrics_canvas.draw()

    def update_info(self):
        # Effacer les anciennes infos
        for i in reversed(range(self.info_layout.count())):
            item = self.info_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)

        # Ajouter les nouvelles infos
        vehicles = traci.vehicle.getIDList()
        self.info_layout.addWidget(QLabel(f"Véhicules actifs: {len(vehicles)}"))
        self.info_layout.addWidget(QLabel(f"Feux contrôlés: {len(traci.trafficlight.getIDList())}"))

        if vehicles:
            sample_veh = vehicles[0]
            speed = traci.vehicle.getSpeed(sample_veh)
            self.info_layout.addWidget(QLabel(f"Exemple - {sample_veh}: {speed:.1f} m/s"))

    def update_q_info(self):
        tl_id = self.tl_combo.currentText()
        if not tl_id:
            return

        q_text = []
        for (tid, state), values in self.sim_thread.q_table.items():
            if tid == tl_id:
                q_text.append(f"État {state}: Maintien={values[0]:.2f}, Changement={values[1]:.2f}")

        self.q_label.setText("\n".join(q_text[:5]) if q_text else "Pas de données pour ce feu")

    def closeEvent(self, event):
        self.sim_thread.stop()
        self.ui_timer.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    config_file = "osm.sumocfg"  # Remplacez par votre fichier de configuration
    dashboard = TrafficDashboard(config_file)
    dashboard.show()
    sys.exit(app.exec_())