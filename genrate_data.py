import csv 
import random 
from datetime import datetime , timedelta


crime_network = {
    "leaders" : ['Xavier' , 'DD' , 'Surya' , 'Elnaaz'] , 
    "lieutenants" : ['Rio' , 'Krystal' , 'Raj' , 'David' , 'Madhav' , 'Ibrahim' , 'Sudhaanshu' , 'Rithika'] , 
    "associates" : ['Sheila' , 'KManohar' , 'Mona' , 'Rohan' , 'Sid' , 'Vatsal' , 'Sam' , 'Akira' , 'Jaanu' , 'Dhanush'] , 
    "external" : ['Abhi' , 'Karan' , 'Monika' ]
}


connection_types = { 
    "phone_call" : (5,15) , 
    "meeting" : (1,8) , 
    "money_transfer" : (10,25) , 
    "email" :(1,5) ,
    "social_media" : (10 ,35) , 
    "family_relation" : (1,1) , 
    "business_meeting" : (2,10)  
}

def calculate_strength(person1, person2, conn_type, frequency):
    """Calculate connection strength based on multiple factors"""
    base_strength = 0.0
    
    # Factor 1: Hierarchy level relationship
    def get_level(person):
        if person in crime_network["leaders"]: return 4
        elif person in crime_network["lieutenants"]: return 3
        elif person in crime_network["associates"]: return 2
        elif person in crime_network["external"]: return 1
        return 0
    
    level1, level2 = get_level(person1), get_level(person2)
    
    # Same level = stronger connection
    if level1 == level2:
        base_strength = 0.7
    # Adjacent levels = medium strength  
    elif abs(level1 - level2) == 1:
        base_strength = 0.5
    # Distant levels = weaker
    else:
        base_strength = 0.3
    
    # Factor 2: Connection type weight
    type_weights = {
        "money_transfer": 0.3,     # High risk = strong connection
        "meeting": 0.2,            # Face-to-face = strong
        "family_relation": 0.4,    # Family = strongest
        "phone_call": 0.15,        # Common = medium
        "business_meeting": 0.25,  # Formal = strong
        "email": 0.1,              # Digital = weaker
        "social_media": 0.05       # Public = weakest
    }
    
    # Factor 3: Frequency normalization (higher freq = stronger)
    freq_bonus = min(frequency / 20.0, 0.3) 
    
    # Calculate final strength
    final_strength = base_strength + type_weights.get(conn_type, 0.1) + freq_bonus
    
    # Ensure it stays within bounds [0.1, 1.0]
    return max(0.1, min(1.0, final_strength))


def generate_connections () : 
    connections = [] 
    all_people = []

    for category in crime_network.values():
        all_people.extend(category)

    for i, leader1 in enumerate(crime_network["leaders"]):
        for leader2 in crime_network["leaders"][i+1:]:
            conn_type = random.choice(["meeting", "phone_call", "business_meeting"])
            freq = random.randint(*connection_types[conn_type])
            strength = calculate_strength(leader1, leader2, conn_type, freq)
            connections.append([leader1, leader2, conn_type, freq, strength])
    
    # Leaders to lieutenants (hierarchical structure)
    for leader in crime_network["leaders"]:
        # Each leader connects to 2-3 lieutenants
        connected_lts = random.sample(crime_network["lieutenants"], random.randint(2, 3))
        for lt in connected_lts:
            conn_type = random.choice(["phone_call", "meeting", "money_transfer"])
            freq = random.randint(*connection_types[conn_type])
            strength = calculate_strength(leader, lt, conn_type, freq)
            connections.append([leader, lt, conn_type, freq, strength])
    
    # Lieutenants to associates
    for lt in crime_network["lieutenants"]:
        # Each lieutenant connects to 2-4 associates
        connected_assocs = random.sample(crime_network["associates"], random.randint(2, 4))
        for assoc in connected_assocs:
            conn_type = random.choice(["phone_call", "meeting", "money_transfer"])
            freq = random.randint(*connection_types[conn_type])
            strength = calculate_strength(lt, assoc, conn_type, freq)
            connections.append([lt, assoc, conn_type, freq, strength])
    
    # Some cross-connections (creates interesting graph structure)
    # Random connections between associates (creates clusters)
    for _ in range(8):  # 8 random associate connections
        person1, person2 = random.sample(crime_network["associates"], 2)
        conn_type = random.choice(["phone_call", "social_media", "meeting"])
        freq = random.randint(*connection_types[conn_type])
        strength = calculate_strength(person1, person2, conn_type, freq)
        connections.append([person1, person2, conn_type, freq, strength])
    
    # External connections (money laundering, legal advice, etc.)
    for ext in crime_network["external"]:
        # Connect to 1-2 leaders
        connected_leaders = random.sample(crime_network["leaders"], random.randint(1, 2))
        for leader in connected_leaders:
            conn_type = random.choice(["business_meeting", "email", "phone_call"])
            freq = random.randint(*connection_types[conn_type])
            strength = calculate_strength(leader, ext, conn_type, freq)
            connections.append([leader, ext, conn_type, freq, strength])

    family_pairs  = {
        ("David" , "Krystal") , 
        ("Madhav" , "Sudhaanshu") ,
        ("Sid" , "Raj") 
    }

    for person1, person2 in family_pairs:
        connections.append([person1, person2, "family_relation", 1, 0.9])
        # Family members also have regular contact
        connections.append([person1, person2, "phone_call", 
                          random.randint(10, 20), 0.7])
    
    return connections

def add_dates_to_connections(connections):
    
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 3, 31)
    
    for conn in connections:
        # Random start date
        days_range = (end_date - start_date).days
        random_start = start_date + timedelta(days=random.randint(0, days_range-30))
        
        # End date is after start date
        random_end = random_start + timedelta(days=random.randint(7, 60))
        if random_end > end_date:
            random_end = end_date
            
        conn.extend([
            random_start.strftime("%Y-%m-%d"),
            random_end.strftime("%Y-%m-%d")
        ])
    
    return connections

def save_to_csv(connections, filename="crime_network.csv"):
    """Save connections to CSV file"""
    headers = ["person1", "person2", "connection_type", "frequency", "strength", "date_start", "date_end"]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(connections)
    
    print(f"Generated {len(connections)} connections")
    print(f"Dataset saved to {filename}")

def generate_node_attributes(filename="crime_nodes.csv"):
    """Generate additional node attributes"""
    all_people = []
    for category_name, people in crime_network.items():
        for person in people:
            role = category_name.rstrip('s')  # remove 's' from category names
            risk_level = {
                "leader": "HIGH",
                "lieutenant": "MEDIUM",
                "associate": "LOW", 
                "external": "MEDIUM"
            }[role]
            
            all_people.append([person, role, risk_level, random.randint(25, 55)])
    
    headers = ["name", "role", "risk_level", "age"]
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(all_people)
    
    print(f"Node attributes saved to {filename}")

if __name__ == "__main__":
    print("Generating Crime Investigation Network Dataset...")
    
    connections = generate_connections()
    connections_with_dates = add_dates_to_connections(connections)
    
    save_to_csv(connections_with_dates)
    generate_node_attributes()
    
    print("\nDataset Summary:")
    print(f"Total People: {sum(len(people) for people in crime_network.values())}")
    print(f"Total Connections: {len(connections)}")
    print("\nNetwork Structure:")
    for category, people in crime_network.items():
        print(f"  {category.title()}: {len(people)} people")
    
    print("\nFiles generated:")
    print("  - crime_network.csv (connections)")
    print("  - crime_nodes.csv (node attributes)")