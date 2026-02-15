# Multi-Drone Forest Fire Detection & Coordination

A multi-agent planning framework for resource-constrained autonomous drones operating in a dynamic forest fire environment.

This project simulates cooperative drone coordination for fire detection and suppression under battery, range, and environmental constraints.

---

## Overview

The system models a fleet of autonomous drones deployed over a discrete grid-based forest environment where fires can spread stochastically over time.

Each drone operates under:

- Limited battery capacity  
- Communication range constraints  
- Return-to-base requirements  
- Dynamic fire propagation  

Decision-making is handled through a Multi-Level Action Tree Rollout (MLAT-R) strategy, enabling forward planning under uncertainty.

---

## Problem Setting

- Grid-based environment  
- Dynamic fire spread model  
- Multi-agent coordination  
- Resource-constrained exploration  
- Real-time replanning  

The objective is to maximize fire detection and containment efficiency while maintaining fleet survivability and energy constraints.

---

## Decision Framework

The MLAT-R strategy evaluates candidate actions using a tree-based rollout mechanism considering:

- Fire proximity and priority  
- Battery thresholds and safe return margins  
- Exploration of unvisited regions  
- Cooperative area coverage  
- Dynamic environmental updates  

This enables structured planning rather than purely reactive behavior.

---

## System Characteristics

- Fully distributed agents  
- Shared situational awareness via communication  
- Resource-aware planning  
- Constraint-based action filtering  
- Swarm-level behavior analysis  

---

## Technical Stack

- Python  
- Multi-agent simulation  
- Graph-based planning  
- Tree rollout evaluation  
- Logging & visualization tools  

---

## Future Extensions

- Integration with learning-based policy refinement  
- Continuous-space environment modeling  
- Probabilistic fire spread forecasting  
- Hybrid planning + reinforcement learning architecture  

---
