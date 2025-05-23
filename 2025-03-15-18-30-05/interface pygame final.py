# -*- coding: utf-8 -*-
"""
Created on Fri May 16 00:29:23 2025

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 15 19:35:31 2025

@author: user
"""

import pygame
import sys
import traci
import random
import numpy as np
from collections import defaultdict, deque
import math
from pygame.locals import *

# Configuration de Pygame
pygame.init()
info = pygame.display.Info()
SCREEN_WIDTH, SCREEN_HEIGHT = info.current_w - 100, info.current_h - 100
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Dashboard de Contrôle RL des Feux de Signalisation")

# Couleurs
BACKGROUND = (240, 240, 240)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
YELLOW = (255, 255, 50)
BLUE = (50, 50, 255)
LIGHT_BLUE = (173, 216, 230)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)

# Polices
font_small = pygame.font.SysFont('Arial', 12)
font_medium = pygame.font.SysFont('Arial', 14)
font_large = pygame.font.SysFont('Arial', 18, bold=True)
font_title = pygame.font.SysFont('Arial', 24, bold=True)

class TrafficLightRL:
    def __init__(self, config_file):
        self.config_file = config_file
        self.running = False
        self.paused = False
        self.speed = 1
        self.q_table = defaultdict(lambda: [0, 0])
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.action_count = defaultdict(int)
        
        # Données pour visualisation
        self.vehicle_history = defaultdict(lambda: deque(maxlen=10))
        self.congestion_data = deque(maxlen=100)
        self.reward_data = deque(maxlen=100)
        self.selected_tl = None
    
    def start_simulation(self):
        traci.start(["sumo-gui", "-c", self.config_file, "--start", "--quit-on-end"])
        self.running = True
        self.selected_tl = traci.trafficlight.getIDList()[0] if traci.trafficlight.getIDList() else None
    
    def stop_simulation(self):
        self.running = False
        traci.close()
    
    def step(self):
        if not self.running or self.paused:
            return
        
        for _ in range(self.speed):
            traci.simulationStep()
            self.run_qlearning_step()
            self.collect_visualization_data()
    
    def run_qlearning_step(self):
        for tl_id in traci.trafficlight.getIDList():
            state = self.get_state(tl_id)
            action = self.choose_action(tl_id, state)
            self.apply_action(tl_id, action)
            next_state = self.get_state(tl_id)
            reward = self.get_reward(tl_id)
            self.update_q_table(tl_id, state, action, reward, next_state)
            
            # Enregistrer l'action pour visualisation
            self.action_count[action] += 1
    
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
    
    def collect_visualization_data(self):
        # Historique des véhicules
        for veh_id in traci.vehicle.getIDList():
            self.vehicle_history[veh_id].append(traci.vehicle.getPosition(veh_id))
        
        # Données de congestion
        congestion = sum(traci.lane.getLastStepHaltingNumber(lane)
                         for tl in traci.trafficlight.getIDList()
                         for lane in traci.trafficlight.getControlledLanes(tl))
        self.congestion_data.append(congestion)
        
        # Données de récompense
        total_reward = sum(self.get_reward(tl_id) for tl_id in traci.trafficlight.getIDList())
        self.reward_data.append(total_reward)

class Dashboard:
    def __init__(self, rl_controller):
        self.rl = rl_controller
        self.show_trajectories = True
        self.show_vehicle_ids = True
        self.show_congestion = True
        self.layout = "horizontal"  # or "vertical"
        
        # UI elements
        self.buttons = []
        self.sliders = []
        self.checkboxes = []
        self.dropdowns = []
        
        self.init_ui()
    
    def init_ui(self):
        # Créer les éléments UI
        self.create_buttons()
        self.create_sliders()
        self.create_checkboxes()
    
    def create_buttons(self):
        # Bouton Start/Pause
        start_btn = {
            "rect": pygame.Rect(20, 20, 120, 40),
            "color": GREEN,
            "text": "Démarrer",
            "action": self.toggle_simulation
        }
        self.buttons.append(start_btn)
        
        # Bouton Paramètres
        settings_btn = {
            "rect": pygame.Rect(160, 20, 120, 40),
            "color": BLUE,
            "text": "Paramètres",
            "action": self.open_settings
        }
        self.buttons.append(settings_btn)
    
    def create_sliders(self):
        # Slider vitesse
        speed_slider = {
            "rect": pygame.Rect(20, 80, 200, 20),
            "min": 1,
            "max": 10,
            "value": 1,
            "label": "Vitesse Simulation"
        }
        self.sliders.append(speed_slider)
    
    def create_checkboxes(self):
        # Checkbox trajectoires
        trajectory_cb = {
            "rect": pygame.Rect(20, 120, 20, 20),
            "checked": True,
            "label": "Afficher trajectoires",
            "action": self.toggle_trajectories
        }
        self.checkboxes.append(trajectory_cb)
        
        # Checkbox IDs véhicules
        ids_cb = {
            "rect": pygame.Rect(20, 150, 20, 20),
            "checked": True,
            "label": "Afficher IDs véhicules",
            "action": self.toggle_vehicle_ids
        }
        self.checkboxes.append(ids_cb)
    
    def toggle_simulation(self):
        if not self.rl.running:
            self.rl.start_simulation()
            self.buttons[0]["text"] = "Pause"
            self.buttons[0]["color"] = YELLOW
        else:
            self.rl.paused = not self.rl.paused
            self.buttons[0]["text"] = "Reprendre" if self.rl.paused else "Pause"
    
    def open_settings(self):
        # Ici vous pourriez ouvrir un panneau de paramètres plus complet
        print("Ouvrir les paramètres")
    
    def toggle_trajectories(self, checked):
        self.show_trajectories = checked
    
    def toggle_vehicle_ids(self, checked):
        self.show_vehicle_ids = checked
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            elif event.type == VIDEORESIZE:
                global SCREEN_WIDTH, SCREEN_HEIGHT
                SCREEN_WIDTH, SCREEN_HEIGHT = event.size
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
                self.init_ui()  # Recalculer les positions des éléments UI
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Clic gauche
                    self.handle_click(event.pos)
        
        return True
    
    def handle_click(self, pos):
        # Vérifier les boutons
        for btn in self.buttons:
            if btn["rect"].collidepoint(pos):
                btn["action"]()
                return
        
        # Vérifier les checkboxes
        for cb in self.checkboxes:
            if cb["rect"].collidepoint(pos):
                cb["checked"] = not cb["checked"]
                cb["action"](cb["checked"])
                return
        
        # Vérifier les sliders
        for slider in self.sliders:
            if slider["rect"].collidepoint(pos):
                # Mettre à jour la valeur du slider
                value_range = slider["max"] - slider["min"]
                pos_in_slider = pos[0] - slider["rect"].x
                slider["value"] = slider["min"] + int((pos_in_slider / slider["rect"].width) * value_range)
                self.rl.speed = slider["value"]
                return
    
    def draw(self):
        screen.fill(BACKGROUND)
        
        # Dessiner la carte de trafic en premier
        self.draw_traffic_map()
        
        # Dessiner les éléments UI par-dessus
        self.draw_sidebar()
        self.draw_metrics()
        
        pygame.display.flip()
    
    def draw_sidebar(self):
        sidebar_width = 300
        sidebar_rect = pygame.Rect(SCREEN_WIDTH - sidebar_width, 0, sidebar_width, SCREEN_HEIGHT)
        pygame.draw.rect(screen, WHITE, sidebar_rect)
        pygame.draw.rect(screen, BLACK, sidebar_rect, 2)
        
        # Titre
        title = font_title.render("Contrôle RL", True, BLACK)
        screen.blit(title, (SCREEN_WIDTH - sidebar_width + 20, 20))
        
        # Dessiner les boutons
        for btn in self.buttons:
            pygame.draw.rect(screen, btn["color"], btn["rect"])
            pygame.draw.rect(screen, BLACK, btn["rect"], 2)
            text = font_medium.render(btn["text"], True, BLACK)
            text_rect = text.get_rect(center=btn["rect"].center)
            screen.blit(text, text_rect)
        
        # Dessiner les sliders
        for slider in self.sliders:
            pygame.draw.rect(screen, GRAY, slider["rect"])
            pygame.draw.rect(screen, BLACK, slider["rect"], 1)
            
            # Dessiner la valeur actuelle
            value_pos = ((slider["value"] - slider["min"]) / (slider["max"] - slider["min"])) * slider["rect"].width
            pygame.draw.rect(screen, BLUE, (slider["rect"].x, slider["rect"].y, value_pos, slider["rect"].height))
            
            # Texte du label
            label = font_small.render(f"{slider['label']}: {slider['value']}", True, BLACK)
            screen.blit(label, (slider["rect"].x, slider["rect"].y - 20))
        
        # Dessiner les checkboxes
        for cb in self.checkboxes:
            pygame.draw.rect(screen, WHITE, cb["rect"])
            pygame.draw.rect(screen, BLACK, cb["rect"], 1)
            if cb["checked"]:
                pygame.draw.line(screen, GREEN, (cb["rect"].x, cb["rect"].y), 
                                (cb["rect"].x + cb["rect"].width, cb["rect"].y + cb["rect"].height), 2)
                pygame.draw.line(screen, GREEN, (cb["rect"].x, cb["rect"].y + cb["rect"].height), 
                                (cb["rect"].x + cb["rect"].width, cb["rect"].y), 2)
            
            # Texte du label
            label = font_small.render(cb["label"], True, BLACK)
            screen.blit(label, (cb["rect"].x + 30, cb["rect"].y))
        
        # Afficher les informations Q-learning
        if self.rl.running and self.rl.selected_tl:
            q_info = self.get_q_info()
            y_pos = 200
            for line in q_info.split('\n'):
                text = font_small.render(line, True, BLACK)
                screen.blit(text, (SCREEN_WIDTH - sidebar_width + 20, y_pos))
                y_pos += 20
    
    def draw_traffic_map(self):
        map_width = SCREEN_WIDTH - 320
        map_height = SCREEN_HEIGHT - 200
        map_rect = pygame.Rect(20, 20, map_width, map_height)
        
        # Créer une surface pour la carte avec transparence
        map_surface = pygame.Surface((map_width, map_height), pygame.SRCALPHA)
        map_surface.fill((240, 240, 240, 0))  # Fond transparent
        
        # Dessiner le cadre de la carte
        pygame.draw.rect(map_surface, BLACK, (0, 0, map_width, map_height), 2)
        
        if self.rl.running:
            try:
                # Dessiner les véhicules
                for veh_id, positions in self.rl.vehicle_history.items():
                    if positions:
                        # Convertir les coordonnées SUMO en coordonnées écran
                        screen_positions = []
                        for x, y in positions:
                            sx = (x / 1000) * (map_width - 40)
                            sy = (y / 1000) * (map_height - 40)
                            screen_positions.append((sx, sy))
                        
                        # Dessiner les trajectoires
                        if self.show_trajectories and len(screen_positions) > 1:
                            pygame.draw.lines(map_surface, LIGHT_BLUE, False, screen_positions, 1)
                        
                        # Dessiner le véhicule
                        if screen_positions:  # Vérifier que la liste n'est pas vide
                            last_pos = screen_positions[-1]
                            pygame.draw.circle(map_surface, BLUE, (int(last_pos[0]), int(last_pos[1])), 4)
                            
                            # Dessiner l'ID si activé
                            if self.show_vehicle_ids:
                                text = font_small.render(veh_id, True, BLACK)
                                map_surface.blit(text, (last_pos[0] + 5, last_pos[1] - 5))

                # Dessiner les feux de signalisation
                for tl_id in traci.trafficlight.getIDList():
                    try:
                        pos = traci.junction.getPosition(tl_id)
                        sx = (pos[0] / 1000) * (map_width - 40)
                        sy = (pos[1] / 1000) * (map_height - 40)
                        
                        # Obtenir l'état actuel du feu
                        state = traci.trafficlight.getRedYellowGreenState(tl_id)
                        color = RED if 'r' in state.lower() else GREEN
                        
                        # Dessiner le feu
                        pygame.draw.circle(map_surface, color, (int(sx), int(sy)), 8)
                        
                        # Dessiner l'ID
                        text = font_small.render(tl_id, True, BLACK)
                        map_surface.blit(text, (sx + 10, sy - 5))
                        
                        # Mettre en évidence le feu sélectionné
                        if tl_id == self.rl.selected_tl:
                            pygame.draw.circle(map_surface, YELLOW, (int(sx), int(sy)), 12, 2)
                    except:
                        continue
                        
            except:
                # Gérer les erreurs de connexion avec TraCI
                error_text = font_medium.render("En attente de connexion SUMO...", True, RED)
                map_surface.blit(error_text, (map_width//2 - 100, map_height//2 - 10))
        
        # Dessiner la surface de la carte sur l'écran principal
        screen.blit(map_surface, (20, 20))
    
    def draw_metrics(self):
        metrics_y = SCREEN_HEIGHT - 170
        metrics_width = SCREEN_WIDTH - 320
        metrics_height = 150
        
        # Graphique de congestion
        congestion_rect = pygame.Rect(20, metrics_y, metrics_width // 2 - 10, metrics_height)
        pygame.draw.rect(screen, WHITE, congestion_rect)
        pygame.draw.rect(screen, BLACK, congestion_rect, 2)
        
        if len(self.rl.congestion_data) > 1:
            max_congestion = max(self.rl.congestion_data) if max(self.rl.congestion_data) > 0 else 1
            points = []
            for i, val in enumerate(self.rl.congestion_data):
                x = 20 + i * (metrics_width // 2 - 30) / len(self.rl.congestion_data)
                y = metrics_y + metrics_height - 20 - (val / max_congestion) * (metrics_height - 40)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(screen, RED, False, points, 2)
        
        # Titre
        title = font_medium.render("Congestion Totale", True, BLACK)
        screen.blit(title, (30, metrics_y + 10))
        
        # Graphique de récompense
        reward_rect = pygame.Rect(metrics_width // 2 + 20, metrics_y, metrics_width // 2 - 10, metrics_height)
        pygame.draw.rect(screen, WHITE, reward_rect)
        pygame.draw.rect(screen, BLACK, reward_rect, 2)
        
        if len(self.rl.reward_data) > 1:
            min_reward = min(self.rl.reward_data)
            max_reward = max(self.rl.reward_data) if max(self.rl.reward_data) != min_reward else min_reward + 1
            points = []
            for i, val in enumerate(self.rl.reward_data):
                x = metrics_width // 2 + 20 + i * (metrics_width // 2 - 30) / len(self.rl.reward_data)
                y = metrics_y + metrics_height - 20 - ((val - min_reward) / (max_reward - min_reward)) * (metrics_height - 40)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(screen, GREEN, False, points, 2)
        
        # Titre
        title = font_medium.render("Récompense Totale", True, BLACK)
        screen.blit(title, (metrics_width // 2 + 30, metrics_y + 10))
    
    def get_q_info(self):
        if not self.rl.selected_tl:
            return "Aucun feu sélectionné"
        
        q_text = [f"Feu: {self.rl.selected_tl}", ""]
        for (tl_id, state), values in self.rl.q_table.items():
            if tl_id == self.rl.selected_tl:
                q_text.append(f"État {state}: Maintien={values[0]:.2f}, Changement={values[1]:.2f}")
        
        return "\n".join(q_text[:8]) if len(q_text) > 2 else "Pas de données pour ce feu"

def main():
    config_file = "osm.sumocfg"  # Remplacez par votre fichier de configuration
    
    # Initialiser le contrôleur RL
    rl_controller = TrafficLightRL(config_file)
    
    # Initialiser le dashboard
    dashboard = Dashboard(rl_controller)
    
    # Boucle principale
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Gérer les événements
        running = dashboard.handle_events()
        
        # Mettre à jour la simulation
        rl_controller.step()
        
        # Dessiner le dashboard
        dashboard.draw()
        
        # Limiter le framerate
        clock.tick(30)
    
    # Arrêter la simulation proprement
    rl_controller.stop_simulation()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()