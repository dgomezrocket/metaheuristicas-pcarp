import osmnx as ox
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

def initialize_routes_nearest_neighbor(graph, num_vehicles, vehicle_capacity):
    """Inicializa rutas usando el algoritmo de Vecino Más Cercano."""
    routes = [[] for _ in range(num_vehicles)]
    vehicle_loads = [0] * num_vehicles
    unvisited_edges = list(graph.edges(keys=True, data=True))

    start_nodes = random.sample(list(graph.nodes), num_vehicles)

    for vehicle_id, start_node in enumerate(start_nodes):
        current_node = start_node
        while unvisited_edges:
            # Obtener aristas desde el nodo actual
            edges_from_node = [edge for edge in unvisited_edges if edge[0] == current_node or edge[1] == current_node]

            # Filtrar aristas que cumplan con la capacidad del vehículo
            valid_edges = []
            for edge in edges_from_node:
                u, v, key, data = edge
                demand = data.get('demand', 1)
                if vehicle_loads[vehicle_id] + demand <= vehicle_capacity:
                    valid_edges.append(edge)

            if not valid_edges:
                break

            # Seleccionar la arista más cercana
            nearest_edge = min(valid_edges, key=lambda edge: edge[3]['length'])
            unvisited_edges.remove(nearest_edge)

            u, v, key, data = nearest_edge
            vehicle_loads[vehicle_id] += data['demand']

            if not routes[vehicle_id]:
                routes[vehicle_id].append((u, v))
                current_node = v
            else:
                last_node = routes[vehicle_id][-1][1]
                if u == last_node:
                    routes[vehicle_id].append((u, v))
                    current_node = v
                else:
                    routes[vehicle_id].append((v, u))
                    current_node = u

    return routes, vehicle_loads, unvisited_edges

def calculate_total_cost(routes, graph):
    """Calcula el costo total de las rutas."""
    total_cost = 0
    for route in routes:
        for u, v in route:
            if graph.has_edge(u, v):
                edge_data = graph.get_edge_data(u, v)
                total_cost += edge_data[0]['length']
    return total_cost

def assign_unvisited_edges(graph, routes, vehicle_loads, unvisited_edges, vehicle_capacity):
    """Asigna las aristas no visitadas a los vehículos disponibles."""
    for edge in unvisited_edges:
        u, v, key, data = edge
        demand = data['demand']

        # Buscar el vehículo más cercano con capacidad disponible
        min_distance = float('inf')
        best_vehicle = None
        best_insert_pos = None

        for vehicle_id, route in enumerate(routes):
            if vehicle_loads[vehicle_id] + demand <= vehicle_capacity:
                for idx, (node_u, node_v) in enumerate(route):
                    distance_to_u = nx.shortest_path_length(graph, source=node_u, target=u, weight='length')
                    distance_to_v = nx.shortest_path_length(graph, source=node_v, target=v, weight='length')

                    if distance_to_u < min_distance:
                        min_distance = distance_to_u
                        best_vehicle = vehicle_id
                        best_insert_pos = idx

                    if distance_to_v < min_distance:
                        min_distance = distance_to_v
                        best_vehicle = vehicle_id
                        best_insert_pos = idx

        if best_vehicle is not None:
            # Insertar la arista en la mejor posición encontrada
            routes[best_vehicle].insert(best_insert_pos + 1, (u, v))
            vehicle_loads[best_vehicle] += demand

    return routes, vehicle_loads

def simulated_annealing(graph, num_vehicles, vehicle_capacity, initial_temp=1000, cooling_rate=0.95, min_temp=1):
    """Algoritmo de Simulated Annealing para PCARP."""
    current_routes, vehicle_loads, unvisited_edges = initialize_routes_nearest_neighbor(graph, num_vehicles, vehicle_capacity)
    current_cost = calculate_total_cost(current_routes, graph)

    # Asignar aristas no visitadas inicialmente
    current_routes, vehicle_loads = assign_unvisited_edges(graph, current_routes, vehicle_loads, unvisited_edges, vehicle_capacity)
    best_routes = current_routes[:]
    best_cost = current_cost

    temp = initial_temp

    while temp > min_temp:
        new_routes, new_vehicle_loads = generate_neighbor(current_routes, graph, vehicle_capacity, vehicle_loads)
        new_cost = calculate_total_cost(new_routes, graph)

        # Aceptar la nueva solución si es mejor o por probabilidad
        if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temp):
            current_routes = new_routes
            vehicle_loads = new_vehicle_loads
            current_cost = new_cost

        # Actualizar la mejor solución encontrada
        if current_cost < best_cost:
            best_routes = current_routes
            best_cost = current_cost

        temp *= cooling_rate

    return best_routes, best_cost

def generate_neighbor(routes, graph, vehicle_capacity, vehicle_loads):
    """Genera una solución vecina con inserciones más inteligentes."""
    new_routes = [route[:] for route in routes]
    new_vehicle_loads = vehicle_loads[:]

    vehicle_id = random.choice([i for i in range(len(new_routes)) if new_routes[i]])

    if len(new_routes[vehicle_id]) > 1:
        idx1, idx2 = random.sample(range(len(new_routes[vehicle_id])), 2)
        new_routes[vehicle_id][idx1], new_routes[vehicle_id][idx2] = new_routes[vehicle_id][idx2], new_routes[vehicle_id][idx1]

    return new_routes, new_vehicle_loads

def plot_vehicle_route(graph, route, vehicle_id):
    """Dibuja la ruta de un solo vehículo sobre el mapa."""
    fig, ax = ox.plot_graph(graph, show=False, close=False, bgcolor='w')
    color = 'r'

    if route:
        for u, v in route:
            if graph.has_edge(u, v):
                x = [graph.nodes[u]['x'], graph.nodes[v]['x']]
                y = [graph.nodes[u]['y'], graph.nodes[v]['y']]
                ax.plot(x, y, color=color, linewidth=3, alpha=0.7)

    plt.title(f"Vehicle {vehicle_id + 1} Route")
    plt.savefig(f'vehicle_{vehicle_id + 1}_route.png')
    plt.show()
    plt.close()

def main():
    city_name = 'Maramburé, Luque, Paraguay'
    graph = ox.graph_from_place(city_name, network_type='drive')
    graph = ox.utils_graph.convert.to_undirected(graph)

    vehicle_capacity = 600  # Capacidad de cada vehículo
    num_vehicles = 4  # Número de vehículos

    for u, v, key, data in graph.edges(keys=True, data=True):
        if 'length' not in data:
            data['length'] = random.uniform(50, 500)  # Longitud aleatoria
        data['demand'] = random.randint(1, 10)  # Demanda aleatoria

    best_routes, best_cost = simulated_annealing(graph, num_vehicles, vehicle_capacity)

    print(f"Best total cost: {best_cost}")
    for i, route in enumerate(best_routes):
        print(f"Vehicle {i + 1} Route: {route}")
        plot_vehicle_route(graph, route, i)

if __name__ == "__main__":
    main()
