import osmnx as ox
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- FUNCIONES PRINCIPALES --------------------------

def initialize_routes(graph, num_vehicles, vehicle_capacity):
    """
    Inicializa rutas para los vehículos utilizando un método básico de inicialización.
    """
    routes = [[] for _ in range(num_vehicles)]
    vehicle_loads = [0] * num_vehicles
    unvisited_edges = list(graph.edges(keys=True, data=True))

    # Inicializa nodos de inicio aleatoriamente para cada vehículo
    start_nodes = random.sample(list(graph.nodes), num_vehicles)

    for vehicle_id, start_node in enumerate(start_nodes):
        current_node = start_node
        while unvisited_edges:
            edges_from_node = [edge for edge in unvisited_edges if edge[0] == current_node or edge[1] == current_node]
            valid_edges = [edge for edge in edges_from_node if vehicle_loads[vehicle_id] + edge[3].get('demand', 1) <= vehicle_capacity]

            if not valid_edges:
                break

            nearest_edge = min(valid_edges, key=lambda edge: edge[3]['length'])
            unvisited_edges.remove(nearest_edge)

            u, v, key, data = nearest_edge
            vehicle_loads[vehicle_id] += data['demand']
            routes[vehicle_id].append((u, v))
            current_node = v if u == current_node else u

    return routes, vehicle_loads

def calculate_total_cost(routes, graph):
    """
    Calcula el costo total de todas las rutas.
    """
    total_cost = 0
    for route in routes:
        for u, v in route:
            if graph.has_edge(u, v):
                edge_data = graph.get_edge_data(u, v)
                total_cost += edge_data[0]['length']
    return total_cost

def vns_algorithm(graph, num_vehicles, vehicle_capacity, max_iterations=100):
    """
    Implementa el algoritmo VNS para el PCARP.
    """
    current_routes, vehicle_loads = initialize_routes(graph, num_vehicles, vehicle_capacity)
    best_routes = current_routes[:]
    best_cost = calculate_total_cost(best_routes, graph)

    for iteration in range(max_iterations):
        k = 1
        while k <= 3:
            new_routes, new_vehicle_loads = shake(current_routes, graph, k)
            new_routes, new_vehicle_loads = local_search(new_routes, graph, vehicle_capacity)

            new_cost = calculate_total_cost(new_routes, graph)

            if new_cost < best_cost:
                best_routes = new_routes
                best_cost = new_cost
                k = 1
            else:
                k += 1

    return best_routes, best_cost

def shake(routes, graph, k):
    """
    Genera una solución aleatoria en el vecindario `k`.
    """
    new_routes = [route[:] for route in routes]

    for _ in range(k):
        vehicle_id = random.choice([i for i in range(len(new_routes)) if new_routes[i]])
        route = new_routes[vehicle_id]
        if len(route) > 1:
            idx1, idx2 = random.sample(range(len(route)), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]

    return new_routes, [sum(graph.get_edge_data(u, v)[0].get('demand', 1) for u, v in route) for route in new_routes]

def local_search(routes, graph, vehicle_capacity):
    """
    Realiza una búsqueda local en la vecindad.
    """
    for vehicle_id, route in enumerate(routes):
        for i in range(len(route) - 1):
            u, v = route[i]
            for j in range(i + 1, len(route)):
                u2, v2 = route[j]
                demand1 = graph.get_edge_data(u, v)[0].get('demand', 1)
                demand2 = graph.get_edge_data(u2, v2)[0].get('demand', 1)

                # Swap si no excede la capacidad
                if demand1 + demand2 <= vehicle_capacity:
                    route[i], route[j] = route[j], route[i]

    return routes, [sum(graph.get_edge_data(u, v)[0].get('demand', 1) for u, v in route) for route in routes]

def plot_all_routes(graph, routes):
    """
    Dibuja todas las rutas generadas en un solo mapa con diferentes colores.
    """
    fig, ax = ox.plot_graph(graph, show=False, close=False, bgcolor='w')
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for vehicle_id, route in enumerate(routes):
        color = colors[vehicle_id % len(colors)]
        for u, v in route:
            if graph.has_edge(u, v):
                x = [graph.nodes[u]['x'], graph.nodes[v]['x']]
                y = [graph.nodes[u]['y'], graph.nodes[v]['y']]
                ax.plot(x, y, color=color, linewidth=3, alpha=0.7)

    plt.title("Vehicle Routes")
    plt.show()
    plt.close()

# -------------------------- EJECUCIÓN PRINCIPAL --------------------------

def main_vns():
    city_name = 'Maramburé, Luque, Paraguay'
    graph = ox.graph_from_place(city_name, network_type='drive')
    graph = ox.utils_graph.convert.to_undirected(graph)

    vehicle_capacity = 600
    num_vehicles = 3

    for u, v, key, data in graph.edges(keys=True, data=True):
        if 'length' not in data:
            data['length'] = random.uniform(50, 500)
        data['demand'] = random.randint(1, 10)

    best_routes, best_cost = vns_algorithm(graph, num_vehicles, vehicle_capacity)
    print(f"Best total cost: {best_cost}")
    for i, route in enumerate(best_routes):
        print(f"Vehicle {i + 1} Route: {route}")

    plot_all_routes(graph, best_routes)

if __name__ == "__main__":
    main_vns()
