import csv
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple
import pandas as pd

class CrimeInvestigationGraph:
    def __init__(self):
        self.nodes = {}  # {name: {role, risk_level, age}}
        self.edges = defaultdict(dict)  # {node1: {node2: {weight, type, frequency}}}
        self.graph_nx = nx.Graph()  # For visualization
        
    def load_data(self, nodes_file='crime_nodes.csv', edges_file='crime_network.csv'):
        """Load crime network data from CSV files"""
        print("Loading crime network data...")

        with open(nodes_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.add_node(
                    row['name'], 
                    role=row['role'],
                    risk_level=row['risk_level'],
                    age=int(row['age'])
                )

        with open(edges_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.add_edge(
                    row['person1'],
                    row['person2'],
                    weight=float(row['strength']),
                    conn_type=row['connection_type'],
                    frequency=int(row['frequency'])
                )
        
        print(f"Loaded {len(self.nodes)} people and {self._count_edges()} connections")
    
    def add_node(self, name: str, **attributes):
        """Add a person to the network"""
        self.nodes[name] = attributes
        self.graph_nx.add_node(name, **attributes)
    
    def add_edge(self, person1: str, person2: str, weight: float, conn_type: str, frequency: int):
        """Add connection between two people"""
        edge_data = {
            'weight': weight,
            'type': conn_type,
            'frequency': frequency
        }
        
        self.edges[person1][person2] = edge_data
        self.edges[person2][person1] = edge_data  # Undirected graph
        
        self.graph_nx.add_edge(person1, person2, 
                              weight=weight, 
                              type=conn_type, 
                              frequency=frequency)
    
    def _count_edges(self):
        """Count total unique edges"""
        count = 0
        for node in self.edges:
            count += len(self.edges[node])
        return count // 2  # Undirected graph
    
    # ============ GRAPH ALGORITHMS ============
    
    def dfs(self, start: str, visited: Set[str] = None) -> List[str]:
        """Depth-First Search traversal"""
        if visited is None:
            visited = set()
        
        if start in visited:
            return []
        
        visited.add(start)
        path = [start]
        
        for neighbor in self.edges[start]:
            if neighbor not in visited:
                path.extend(self.dfs(neighbor, visited))
        
        return path
    
    def bfs(self, start: str) -> List[str]:
        """Breadth-First Search traversal"""
        visited = set()
        queue = deque([start])
        path = []
        
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                path.append(node)
                
                for neighbor in self.edges[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return path
    
    def find_path_bfs(self, start: str, end: str) -> List[str]:
        """Find shortest path between two people using BFS"""
        if start == end:
            return [start]
        
        visited = set()
        queue = deque([(start, [start])])
        
        while queue:
            node, path = queue.popleft()
            
            if node in visited:
                continue
                
            visited.add(node)
            
            for neighbor in self.edges[node]:
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def tarjan_scc(self) -> List[List[str]]:
        """
        Tarjan's Algorithm for finding Strongly Connected Components
        Modified for undirected graph - finds connected components
        """
        visited = set()
        components = []
        
        def dfs_component(node, component):
            visited.add(node)
            component.append(node)
            
            for neighbor in self.edges[node]:
                if neighbor not in visited:
                    dfs_component(neighbor, component)
        
        for node in self.nodes:
            if node not in visited:
                component = []
                dfs_component(node, component)
                if len(component) > 1:  # Only include multi-node components
                    components.append(component)
        
        return components
    
    def find_articulation_points(self) -> Set[str]:
        """Find articulation points (critical people whose removal disconnects network)"""
        visited = set()
        disc = {}  # Discovery time
        low = {}   # Low value
        parent = {}
        ap = set()  # Articulation points
        time = [0] 
        
        def bridge_util(u):
            children = 0
            visited.add(u)
            disc[u] = low[u] = time[0]
            time[0] += 1
            
            for v in self.edges[u]:
                if v not in visited:
                    children += 1
                    parent[v] = u
                    bridge_util(v)
                    
                    low[u] = min(low[u], low[v])
                    
                    # u is articulation point in following cases:
                    # 1) u is root and has more than one child
                    if parent.get(u) is None and children > 1:
                        ap.add(u)
                    
                    # 2) u is not root and low[v] >= disc[u]
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])
        
        for node in self.nodes:
            if node not in visited:
                bridge_util(node)
        
        return ap
    
    def dijkstra_shortest_path(self, start: str, end: str) -> Tuple[List[str], float]:
        """Find shortest weighted path between two people"""
        if start == end:
            return [start], 0.0
        
        distances = {node: float('infinity') for node in self.nodes}
        distances[start] = 0
        previous = {}
        unvisited = set(self.nodes.keys())
        
        while unvisited:
            current = min(unvisited, key=lambda x: distances[x])
            unvisited.remove(current)
            
            if current == end:
                break
            
            for neighbor in self.edges[current]:
                if neighbor in unvisited:
                    # Use inverse of strength as distance (stronger = shorter)
                    weight = 1.0 / self.edges[current][neighbor]['weight']
                    alt = distances[current] + weight
                    
                    if alt < distances[neighbor]:
                        distances[neighbor] = alt
                        previous[neighbor] = current
        
        # Reconstruct path
        if end not in previous and end != start:
            return [], float('infinity')
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous.get(current)
        
        return path[::-1], distances[end]
    
    # ============ ANALYSIS FUNCTIONS ============
    
    def get_network_stats(self) -> Dict:
        """Get basic network statistics"""
        return {
            'total_people': len(self.nodes),
            'total_connections': self._count_edges(),
            'density': (2 * self._count_edges()) / (len(self.nodes) * (len(self.nodes) - 1)),
            'leaders': len([n for n in self.nodes if self.nodes[n]['role'] == 'leader']),
            'lieutenants': len([n for n in self.nodes if self.nodes[n]['role'] == 'lieutenant']),
            'associates': len([n for n in self.nodes if self.nodes[n]['role'] == 'associate']),
            'external': len([n for n in self.nodes if self.nodes[n]['role'] == 'external'])
        }
    
    def analyze_person(self, person: str) -> Dict:
        """Analyze a specific person's network position"""
        if person not in self.nodes:
            return {}
        
        connections = list(self.edges[person].keys())
        strong_connections = [
            p for p in connections 
            if self.edges[person][p]['weight'] > 0.8
        ]
        
        return {
            'role': self.nodes[person]['role'],
            'risk_level': self.nodes[person]['risk_level'],
            'total_connections': len(connections),
            'strong_connections': len(strong_connections),
            'connection_list': connections,
            'avg_connection_strength': sum(
                self.edges[person][p]['weight'] for p in connections
            ) / len(connections) if connections else 0
        }
    
    # ============ VISUALIZATION ============
    
    def visualize_network_static(self, highlight_articulation_points=True):
        """Create static network visualization"""
        plt.figure(figsize=(15, 10))
        
        # Create layout
        pos = nx.spring_layout(self.graph_nx, k=3, iterations=50)
        
        # Color nodes by role
        color_map = {
            'leader': '#FF4444',      # Red
            'lieutenant': '#FF8800',  # Orange  
            'associate': '#4488FF',   # Blue
            'external': '#44FF44'     # Green
        }
        
        node_colors = [color_map[self.nodes[node]['role']] for node in self.graph_nx.nodes()]
        
        # Draw edges with varying thickness based on strength
        edges = self.graph_nx.edges(data=True)
        edge_weights = [edge[2]['weight'] * 3 for edge in edges]  # Scale for visibility
        
        nx.draw_networkx_edges(self.graph_nx, pos, width=edge_weights, alpha=0.6, edge_color='gray')
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph_nx, pos, node_color=node_colors, 
                              node_size=800, alpha=0.9)
        
        # Highlight articulation points
        if highlight_articulation_points:
            ap = self.find_articulation_points()
            if ap:
                nx.draw_networkx_nodes(self.graph_nx, pos, nodelist=list(ap),
                                     node_color='purple', node_size=1000, alpha=0.7)
        
        # Add labels
        nx.draw_networkx_labels(self.graph_nx, pos, font_size=8, font_weight='bold')
        
        plt.title("Crime Investigation Network\nRed=Leaders, Orange=Lieutenants, Blue=Associates, Green=External\nPurple=Articulation Points", 
                 fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self):
        """Create an interactive investigation dashboard"""
        print("\n" + "="*60)
        print("🎮 INTERACTIVE CRIME INVESTIGATION DASHBOARD")
        print("="*60)
        
        while True:
            print("\n📋 INVESTIGATION OPTIONS:")
            print("1. 🔍 Find connection between two people")
            print("2. 👤 Analyze specific person")
            print("3. 🎯 Find all people connected to someone")
            print("4. 💥 Simulate arrest impact (remove person)")
            print("5. 🔗 Find strongest connection path")
            print("6. 📊 Show network statistics")
            print("7. 🌐 Show network visualization")
            print("8. 🚪 Exit investigation")
            
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == '1':
                self._interactive_find_connection()
            elif choice == '2':
                self._interactive_analyze_person()
            elif choice == '3':
                self._interactive_show_connections()
            elif choice == '4':
                self._interactive_simulate_arrest()
            elif choice == '5':
                self._interactive_strongest_path()
            elif choice == '6':
                self._show_stats()
            elif choice == '7':
                self.visualize_network_static()
            elif choice == '8':
                print("🔚 Investigation closed.")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def _interactive_find_connection(self):
        """Interactive connection finder"""
        print("\n🔍 FIND CONNECTION BETWEEN TWO PEOPLE")
        print("Available people:", ", ".join(sorted(self.nodes.keys())))
        
        person1 = input("Enter first person: ").strip()
        person2 = input("Enter second person: ").strip()
        
        if person1 not in self.nodes or person2 not in self.nodes:
            print("❌ One or both people not found in network")
            return
        
        path = self.find_path_bfs(person1, person2)
        if path:
            print(f"✅ Connection found: {' → '.join(path)}")
            print(f"📏 Degrees of separation: {len(path) - 1}")
            
            # Show connection details
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i+1]
                edge_data = self.edges[p1][p2]
                print(f"   {p1} ↔ {p2}: {edge_data['type']} (strength: {edge_data['weight']:.2f})")
        else:
            print("❌ No connection found between these people")
    
    def _interactive_analyze_person(self):
        """Interactive person analysis"""
        print("\n👤 PERSON ANALYSIS")
        print("Available people:", ", ".join(sorted(self.nodes.keys())))
        
        person = input("Enter person to analyze: ").strip()
        
        if person not in self.nodes:
            print("❌ Person not found in network")
            return
        
        analysis = self.analyze_person(person)
        print(f"\n📋 ANALYSIS REPORT FOR: {person}")
        print(f"   Role: {analysis['role'].upper()}")
        print(f"   Risk Level: {analysis['risk_level']}")
        print(f"   Age: {self.nodes[person]['age']}")
        print(f"   Total Connections: {analysis['total_connections']}")
        print(f"   Strong Connections (>0.8): {analysis['strong_connections']}")
        print(f"   Average Connection Strength: {analysis['avg_connection_strength']:.2f}")
        
        print(f"\n🔗 CONNECTED TO:")
        for conn in analysis['connection_list']:
            edge_data = self.edges[person][conn]
            role = self.nodes[conn]['role']
            print(f"   • {conn} ({role}) - {edge_data['type']} (strength: {edge_data['weight']:.2f})")
    
    def _interactive_show_connections(self):
        """Show all connections for a person"""
        print("\n🎯 SHOW ALL CONNECTIONS")
        print("Available people:", ", ".join(sorted(self.nodes.keys())))
        
        person = input("Enter person: ").strip()
        
        if person not in self.nodes:
            print("❌ Person not found in network")
            return
        
        connections = self.bfs(person)
        print(f"\n🌐 All people reachable from {person}:")
        print(f"   Total reachable: {len(connections)}")
        
        by_role = defaultdict(list)
        for conn in connections:
            by_role[self.nodes[conn]['role']].append(conn)
        
        for role, people in by_role.items():
            print(f"   {role.title()}s: {', '.join(people)}")
    
    def _interactive_simulate_arrest(self):
        """Simulate arresting someone and show network impact"""
        print("\n💥 SIMULATE ARREST IMPACT")
        print("Available people:", ", ".join(sorted(self.nodes.keys())))
        
        person = input("Enter person to arrest: ").strip()
        
        if person not in self.nodes:
            print("❌ Person not found in network")
            return
        
        # Count components before
        components_before = self.tarjan_scc()
        total_before = sum(len(comp) for comp in components_before)
        
        # Temporarily remove person
        original_edges = self.edges[person].copy()
        original_node = self.nodes[person].copy()
        
        # Remove from graph
        for neighbor in list(original_edges.keys()):
            del self.edges[neighbor][person]
        del self.edges[person]
        del self.nodes[person]
        self.graph_nx.remove_node(person)
        
        # Count components after
        components_after = self.tarjan_scc()
        total_after = sum(len(comp) for comp in components_after)
        
        print(f"\n📊 ARREST IMPACT ANALYSIS:")
        print(f"   Person arrested: {person} ({original_node['role']})")
        print(f"   Network size before: {len(self.nodes) + 1} people")
        print(f"   Network size after: {len(self.nodes)} people")
        print(f"   Connected people before: {total_before}")
        print(f"   Connected people after: {total_after}")
        print(f"   Network fragmentation: {total_before - total_after} people isolated")
        
        if len(components_after) > len(components_before):
            print(f"   ⚠️  Network broke into {len(components_after)} fragments!")
        else:
            print(f"   ✅ Network remained connected")
        
        # Restore person
        self.nodes[person] = original_node
        self.edges[person] = original_edges
        self.graph_nx.add_node(person, **original_node)
        for neighbor, edge_data in original_edges.items():
            self.edges[neighbor][person] = edge_data
            self.graph_nx.add_edge(person, neighbor, **edge_data)
    
    def _interactive_strongest_path(self):
        """Find strongest connection path between two people"""
        print("\n🔗 FIND STRONGEST CONNECTION PATH")
        print("Available people:", ", ".join(sorted(self.nodes.keys())))
        
        person1 = input("Enter first person: ").strip()
        person2 = input("Enter second person: ").strip()
        
        if person1 not in self.nodes or person2 not in self.nodes:
            print("❌ One or both people not found in network")
            return
        
        path, distance = self.dijkstra_shortest_path(person1, person2)
        if path:
            strength = 1 / distance if distance > 0 else 1.0
            print(f"✅ Strongest path found: {' → '.join(path)}")
            print(f"📊 Overall connection strength: {strength:.3f}")
            
            print(f"\n🔗 Path details:")
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i+1]
                edge_data = self.edges[p1][p2]
                print(f"   {p1} → {p2}: {edge_data['type']} (strength: {edge_data['weight']:.2f})")
        else:
            print("❌ No path found between these people")
    
    def _show_stats(self):
        """Show detailed network statistics"""
        stats = self.get_network_stats()
        print(f"\n📊 DETAILED NETWORK STATISTICS:")
        print(f"   Total People: {stats['total_people']}")
        print(f"   Total Connections: {stats['total_connections']}")
        print(f"   Network Density: {stats['density']:.3f}")
        print(f"   Leaders: {stats['leaders']}")
        print(f"   Lieutenants: {stats['lieutenants']}")
        print(f"   Associates: {stats['associates']}")
        print(f"   External Contacts: {stats['external']}")
        
        # Show articulation points
        ap = self.find_articulation_points()
        print(f"\n🔑 Critical People (Articulation Points): {len(ap)}")
        for person in ap:
            role = self.nodes[person]['role']
            print(f"   • {person} ({role})")
        
        # Show components
        components = self.tarjan_scc()
        print(f"\n🎯 Connected Groups: {len(components)}")
        for i, comp in enumerate(components, 1):
            print(f"   Group {i}: {', '.join(comp[:5])}{'...' if len(comp) > 5 else ''} ({len(comp)} people)")

# ============ DEMO FUNCTIONS ============

def run_investigation_demo():
    """Run a complete crime investigation demo"""
    print("="*50)
    print("🔍 CRIME INVESTIGATION NETWORK ANALYSIS")
    print("="*50)
    
    # Initialize and load data
    crime_net = CrimeInvestigationGraph()
    crime_net.load_data()
    
    print("\n📊 NETWORK STATISTICS:")
    stats = crime_net.get_network_stats()
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🔗 NETWORK DENSITY: {stats['density']:.3f} (How connected is the network)")
    
    print("\n🎯 FINDING CRIMINAL GROUPS (Connected Components):")
    components = crime_net.tarjan_scc()
    for i, component in enumerate(components, 1):
        print(f"  Group {i}: {', '.join(component)}")
    
    print("\n🔑 CRITICAL PEOPLE (Articulation Points):")
    critical_people = crime_net.find_articulation_points()
    for person in critical_people:
        role = crime_net.nodes[person]['role']
        print(f"  {person} ({role}) - Network would fragment if removed")
    
    print("\n🕵️ INVESTIGATION QUERIES:")
    
    # Find connection between two people
    path = crime_net.find_path_bfs("Xavier", "Dhanush")
    if path:
        print(f"  Connection Xavier → Dhanush: {' → '.join(path)}")
    
    # Analyze key person
    print(f"\n📋 ANALYZING KEY SUSPECT: Xavier")
    analysis = crime_net.analyze_person("Xavier")
    print(f"  Role: {analysis['role']}")
    print(f"  Risk Level: {analysis['risk_level']}")
    print(f"  Total Connections: {analysis['total_connections']}")
    print(f"  Strong Connections: {analysis['strong_connections']}")
    print(f"  Average Connection Strength: {analysis['avg_connection_strength']:.2f}")
    
    # Shortest weighted path
    path, distance = crime_net.dijkstra_shortest_path("DD", "Sam")
    if path:
        print(f"\n🎯 STRONGEST CONNECTION PATH DD → Sam:")
        print(f"  Path: {' → '.join(path)}")
        print(f"  Connection Strength: {1/distance:.2f}")
    
    print("\n" + "="*50)
    print("🎮 Starting Interactive Investigation Dashboard...")
    crime_net.create_interactive_dashboard()

if __name__ == "__main__":
    run_investigation_demo()