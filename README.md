# Crime Investigation Network – Advanced Graph Algorithms Project

## Overview

This project models a **crime investigation network** using advanced graph algorithms. It helps investigators analyze criminal organizations by representing individuals as nodes and their relationships as edges in a graph. The system can:

- Identify **critical links and people** (articulation points)
- **Visualize** the network (static and interactive)
- **Analyze connections and paths** (BFS, DFS, Dijkstra, connected components)
- **Simulate arrests** and their impact on the network
- Provide an **interactive dashboard** for hands-on investigation

---

## Features

- **Data Generation**: Generate synthetic CSV data for nodes and edges using `genrate_data.py`, or use the provided `crime_nodes.csv` and `crime_network.csv`.
- **Graph Algorithms**: Includes DFS, BFS, shortest path (Dijkstra), connected components, and articulation point detection.
- **Visualization**: Static and interactive network visualizations using `matplotlib` and `networkx`.
- **Analysis Tools**: Find critical people, analyze individuals, simulate arrests, and more.
- **Interactive Dashboard**: CLI-based dashboard for hands-on investigation.

---

## File Structure

- `crime_investigation.py` – Main logic for graph construction, algorithms, visualization, and dashboard.
- `genrate_data.py` – Script to generate random but realistic crime network data.
- `crime_nodes.csv` – Example node data (name, role, risk level, age).
- `crime_network.csv` – Example edge data (person1, person2, connection type, frequency, strength, date range).

---

## Getting Started

### 1. Install Requirements

Make sure you have Python 3.7+ and install the required packages:

```bash
pip install matplotlib networkx pandas
```

### 2. Generate or Use Provided Data

- To use the provided data, skip to the next step.
- To generate new data:

```bash
python genrate_data.py
```

This will create fresh `crime_nodes.csv` and `crime_network.csv` files.

### 3. Run the Investigation Tool

```bash
python crime_investigation.py
```

This will:
- Print network statistics and analysis
- Show critical people and groups
- Allow you to interactively explore the network, find paths, analyze suspects, and simulate arrests

---

## Example Data

**Nodes (`crime_nodes.csv`):**
```
name,role,risk_level,age
Xavier,leader,HIGH,44
Rio,lieutenant,MEDIUM,52
Sheila,associate,LOW,46
Abhi,external,MEDIUM,45
...
```

**Edges (`crime_network.csv`):**
```
person1,person2,connection_type,frequency,strength,date_start,date_end
Xavier,DD,meeting,6,1.0,2025-02-08,2025-03-22
Krystal,Mona,meeting,3,0.85,2025-01-04,2025-02-28
...
```

---

## Main Functionalities

- **Network Statistics**: Total people, connections, density, breakdown by role.
- **Critical People**: Articulation points whose removal fragments the network.
- **Connected Groups**: Find criminal subgroups (connected components).
- **Path Finding**: Find shortest or strongest connection paths between suspects.
- **Person Analysis**: Role, risk, connections, and strength.
- **Arrest Simulation**: See how removing a person affects the network.
- **Visualization**: Static and interactive network plots.

---

## Customization

- Edit `genrate_data.py` to change the network structure, connection types, or data generation logic.
- Use your own CSVs with the same format for real-world data.

