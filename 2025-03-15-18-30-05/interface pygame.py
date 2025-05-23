# -*- coding: utf-8 -*-
"""
Traffic Light RL Control Dashboard
"""

import traci
import random
import numpy as np
import pygame
import sys
from pygame.locals import *
from collections import deque

# Simulation parameters
config_file = "osm.sumocfg"
simulation_steps = 100000

# RL parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Initialize Q-table
q_table = {}

# Dashboard setup
pygame.init()
pygame.font.init()

# Screen dimensions (adjustable for different screen sizes)
info = pygame.display.Info()
SCREEN_WIDTH = info.current_w - 100
SCREEN_HEIGHT = info.current_h - 100
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Traffic Light RL Control Dashboard")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (173, 216, 230)

# Fonts
title_font = pygame.font.SysFont('Arial', 24, bold=True)
header_font = pygame.font.SysFont('Arial', 18, bold=True)
normal_font = pygame.font.SysFont('Arial', 14)
small_font = pygame.font.SysFont('Arial', 12)

# Dashboard layout parameters
MARGIN = 20
PANEL_WIDTH = SCREEN_WIDTH // 3 - MARGIN * 1.5
PANEL_HEIGHT = SCREEN_HEIGHT - MARGIN * 2

# Data structures for visualization
congestion_history = {}
decision_history = {}

def get_state(tl_id):
    """Get traffic light state (number of waiting vehicles)"""
    state = 0
    for lane in traci.trafficlight.getControlledLanes(tl_id):
        state += traci.lane.getLastStepHaltingNumber(lane)
    return state

def get_reward(tl_id):
    """Calculate reward (negative of waiting vehicles)"""
    reward = 0
    for lane in traci.trafficlight.getControlledLanes(tl_id):
        reward -= traci.lane.getLastStepHaltingNumber(lane)
    return reward

def choose_action(tl_id, state):
    """Choose action using epsilon-greedy policy"""
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1])  # Exploration
    else:
        if (tl_id, state) not in q_table:
            return random.choice([0, 1])
        else:
            return np.argmax(q_table[(tl_id, state)])  # Exploitation

def apply_action(tl_id, action):
    """Apply action (change traffic light state)"""
    current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
    if action == 1:
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
    """Update Q-table"""
    if (tl_id, state) not in q_table:
        q_table[(tl_id, state)] = [0, 0]
    if (tl_id, next_state) not in q_table:
        q_table[(tl_id, next_state)] = [0, 0]

    q_table[(tl_id, state)][action] = q_table[(tl_id, state)][action] + alpha * (
        reward + gamma * np.max(q_table[(tl_id, next_state)]) - q_table[(tl_id, state)][action])

def draw_traffic_light_panel(tl_id, x, y, width, height):
    """Draw traffic light status panel"""
    # Panel background
    pygame.draw.rect(screen, WHITE, (x, y, width, height))
    pygame.draw.rect(screen, BLACK, (x, y, width, height), 2)
    
    # Title
    title = header_font.render(f"Traffic Light: {tl_id}", True, BLACK)
    screen.blit(title, (x + 10, y + 10))
    
    # Current state
    state = get_state(tl_id)
    state_text = normal_font.render(f"Waiting vehicles: {state}", True, BLACK)
    screen.blit(state_text, (x + 10, y + 40))
    
    # Light status visualization
    light_state = traci.trafficlight.getRedYellowGreenState(tl_id)
    light_x = x + width - 60
    for i, s in enumerate(light_state[:4]):  # Show first 4 lights
        color = BLACK
        if s == 'r':
            color = RED
        elif s == 'y':
            color = YELLOW
        elif s == 'g':
            color = GREEN
            
        pygame.draw.circle(screen, color, (light_x, y + 40 + i * 30), 10)
    
    # Congestion graph
    congestion_title = normal_font.render("Congestion History:", True, BLACK)
    screen.blit(congestion_title, (x + 10, y + 80))
    
    if len(congestion_history[tl_id]) > 1:
        max_congestion = max(congestion_history[tl_id]) if max(congestion_history[tl_id]) > 0 else 1
        points = []
        for i, val in enumerate(congestion_history[tl_id]):
            x_pos = x + 10 + i * (width - 20) / len(congestion_history[tl_id])
            y_pos = y + 160 - (val / max_congestion) * 60
            points.append((x_pos, y_pos))
        
        if len(points) > 1:
            pygame.draw.lines(screen, BLUE, False, points, 2)
    
    # Decision history
    decision_title = normal_font.render("Recent Decisions:", True, BLACK)
    screen.blit(decision_title, (x + 10, y + 170))
    
    for i, decision in enumerate(list(decision_history[tl_id])[-5:]):
        action_text = "Changed" if decision else "Maintained"
        color = GREEN if decision else RED
        text = small_font.render(action_text, True, color)
        screen.blit(text, (x + 10, y + 200 + i * 20))

def draw_q_learning_panel(x, y, width, height):
    """Draw Q-learning information panel"""
    # Panel background
    pygame.draw.rect(screen, WHITE, (x, y, width, height))
    pygame.draw.rect(screen, BLACK, (x, y, width, height), 2)
    
    # Title
    title = header_font.render("RL Decision Making", True, BLACK)
    screen.blit(title, (x + 10, y + 10))
    
    # Parameters
    params = [
        f"Learning Rate (α): {alpha}",
        f"Discount Factor (γ): {gamma}",
        f"Exploration Rate (ε): {epsilon}"
    ]
    
    for i, param in enumerate(params):
        text = normal_font.render(param, True, BLACK)
        screen.blit(text, (x + 10, y + 40 + i * 25))
    
    # Q-table info
    q_info = normal_font.render(f"States in Q-table: {len(q_table)}", True, BLACK)
    screen.blit(q_info, (x + 10, y + 120))
    
    # Example Q-values (show first few if available)
    if q_table:
        example = normal_font.render("Example Q-values:", True, BLACK)
        screen.blit(example, (x + 10, y + 150))
        
        for i, ((tl_id, state), actions) in enumerate(list(q_table.items())[:3]):
            text = small_font.render(f"TL {tl_id}, State {state}: {actions}", True, BLUE)
            screen.blit(text, (x + 10, y + 180 + i * 20))

def draw_performance_panel(x, y, width, height):
    """Draw performance metrics panel"""
    # Panel background
    pygame.draw.rect(screen, WHITE, (x, y, width, height))
    pygame.draw.rect(screen, BLACK, (x, y, width, height), 2)
    
    # Title
    title = header_font.render("Performance Metrics", True, BLACK)
    screen.blit(title, (x + 10, y + 10))
    
    # Calculate metrics
    total_waiting = 0
    total_changes = 0
    for tl_id in traci.trafficlight.getIDList():
        total_waiting += get_state(tl_id)
        total_changes += sum(decision_history[tl_id])
    
    # Display metrics
    metrics = [
        f"Total Waiting Vehicles: {total_waiting}",
        f"Total Light Changes: {total_changes}",
        f"Average Congestion: {total_waiting / max(1, len(traci.trafficlight.getIDList())):.1f}"
    ]
    
    for i, metric in enumerate(metrics):
        text = normal_font.render(metric, True, BLACK)
        screen.blit(text, (x + 10, y + 40 + i * 25))
    
    # Learning progress (simple visualization)
    progress_title = normal_font.render("Learning Progress:", True, BLACK)
    screen.blit(progress_title, (x + 10, y + 120))
    
    if q_table:
        # Simple measure of learning progress - average Q-value magnitude
        avg_q = np.mean([np.max(values) for values in q_table.values()])
        progress_width = min(width - 40, avg_q * 10)
        pygame.draw.rect(screen, LIGHT_BLUE, (x + 10, y + 150, progress_width, 20))
        pygame.draw.rect(screen, BLACK, (x + 10, y + 150, width - 40, 20), 1)
        
        progress_text = small_font.render(f"Avg Q: {avg_q:.2f}", True, BLACK)
        screen.blit(progress_text, (x + 10, y + 175))

def draw_dashboard(step):
    """Draw the complete dashboard"""
    screen.fill(GRAY)
    
    # Main title
    title = title_font.render("Traffic Light Reinforcement Learning Control", True, BLACK)
    screen.blit(title, (MARGIN, MARGIN))
    
    # Simulation step
    step_text = normal_font.render(f"Simulation Step: {step}/{simulation_steps}", True, BLACK)
    screen.blit(step_text, (SCREEN_WIDTH - 200, MARGIN))
    
    # Traffic light panels (2 columns)
    tl_ids = traci.trafficlight.getIDList()
    for i, tl_id in enumerate(tl_ids[:4]):  # Show up to 4 traffic lights
        col = i % 2
        row = i // 2
        x = MARGIN + col * (PANEL_WIDTH + MARGIN)
        y = MARGIN + 50 + row * (PANEL_HEIGHT // 2 + MARGIN)
        draw_traffic_light_panel(tl_id, x, y, PANEL_WIDTH, PANEL_HEIGHT // 2 - MARGIN)
    
    # RL panel
    draw_q_learning_panel(SCREEN_WIDTH - PANEL_WIDTH - MARGIN, MARGIN + 50, 
                         PANEL_WIDTH, PANEL_HEIGHT // 2 - MARGIN)
    
    # Performance panel
    draw_performance_panel(SCREEN_WIDTH - PANEL_WIDTH - MARGIN, 
                          MARGIN + 50 + PANEL_HEIGHT // 2, 
                          PANEL_WIDTH, PANEL_HEIGHT // 2 - MARGIN)
    
    pygame.display.flip()

def handle_events():
    """Handle pygame events"""
    for event in pygame.event.get():
        if event.type == QUIT:
            return False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                return False
        elif event.type == VIDEORESIZE:
            global SCREEN_WIDTH, SCREEN_HEIGHT, PANEL_WIDTH, PANEL_HEIGHT
            SCREEN_WIDTH, SCREEN_HEIGHT = event.size
            PANEL_WIDTH = SCREEN_WIDTH // 3 - MARGIN * 1.5
            PANEL_HEIGHT = SCREEN_HEIGHT - MARGIN * 2
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    return True

# Main simulation loop
def run_simulation():
    traci.start(["sumo-gui", "-c", config_file])
    
    # Initialize visualization data structures
    for tl_id in traci.trafficlight.getIDList():
        congestion_history[tl_id] = deque(maxlen=50)
        decision_history[tl_id] = deque(maxlen=50)
    
    running = True
    step = 0
    
    while step < simulation_steps and running:
        running = handle_events()
        if not running:
            break
            
        traci.simulationStep()
        
        # Control traffic lights with Q-learning
        for tl_id in traci.trafficlight.getIDList():
            state = get_state(tl_id)
            action = choose_action(tl_id, state)
            apply_action(tl_id, action)
            next_state = get_state(tl_id)
            reward = get_reward(tl_id)
            update_q_table(tl_id, state, action, reward, next_state)
            
            # Update visualization data
            congestion_history[tl_id].append(state)
            decision_history[tl_id].append(action)
        
        # Update dashboard every 10 steps for better performance
        if step % 10 == 0:
            draw_dashboard(step)
        
        step += 1
    
    traci.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_simulation()