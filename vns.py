import osmnx as ox
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from sklearn.cluster import KMeans

# -------------------------- FUNCIONES PRINCIPALES --------------------------

def cluster_edges(graph, num_clusters):
    """
    Agrupa las aristas del grafo en clusters geográficos.

    Parámetros:
    - graph: Grafo de la ciudad.
    - num_clusters: Número de clusters o áreas.

    Retorna:
    - edge_clusters: Diccionario que asigna cada arista a un cluster.
    """
    edges = list(graph.edges(data=True))
    edge_coords = []

    for u, v, data in edges:
        x = (graph.nodes[u]['x'] + graph.nodes[v]['x']) / 2
        y = (graph.nodes[u]['y'] + graph.nodes[v]['y']) / 2
        edge_coords.append([x, y])

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(edge_coords)
    labels = kmeans.labels_

    edge_clusters = {}
    for idx, (u, v, data) in enumerate(edges):
        edge_clusters[(u, v)] = labels[idx]

    return edge_clusters

def initialize_routes_clustered(graph, num_vehicles, vehicle_capacity):
    """
    Inicializa rutas asignando todas las aristas a los vehículos basándose en clusters.

    Retorna:
    - routes: Lista de rutas para cada vehículo.
    - vehicle_loads: Lista de cargas actuales de cada vehículo.
    """
    routes = [[] for _ in range(num_vehicles)]
    vehicle_loads = [0] * num_vehicles

    # Agrupar aristas en clusters
    edge_clusters = cluster_edges(graph, num_vehicles)

    # Crear una lista de aristas con demanda
    demand_edges = [(u, v, data) for u, v, data in graph.edges(data=True) if data.get('demand', 0) > 0]

    # Asignar aristas a vehículos según el cluster
    for (u, v, data) in demand_edges:
        cluster = edge_clusters.get((u, v)) or edge_clusters.get((v, u))
        vehicle_id = cluster % num_vehicles

        demand = data.get('demand', 1)
        if vehicle_loads[vehicle_id] + demand <= vehicle_capacity:
            routes[vehicle_id].append((u, v))
            vehicle_loads[vehicle_id] += demand
        else:
            # Intentar asignar a otro vehículo con capacidad disponible
            assigned = False
            for vid in range(num_vehicles):
                if vehicle_loads[vid] + demand <= vehicle_capacity:
                    routes[vid].append((u, v))
                    vehicle_loads[vid] += demand
                    assigned = True
                    break
            if not assigned:
                print(f"No se pudo asignar la arista ({u}, {v}) debido a restricciones de capacidad.")

    # Asegurar que las rutas sean conectadas
    for vehicle_id in range(num_vehicles):
        route = routes[vehicle_id]
        if not route:
            continue
        # Construir subgrafo de la ruta
        subgraph = nx.Graph()
        for u, v in route:
            subgraph.add_edge(u, v)
        # Conectar componentes si es necesario
        if not nx.is_connected(subgraph):
            components = list(nx.connected_components(subgraph))
            while len(components) > 1:
                comp1 = components[0]
                comp2 = components[1]
                min_distance = float('inf')
                closest_nodes = None
                for node_u in comp1:
                    for node_v in comp2:
                        try:
                            distance = nx.shortest_path_length(graph, source=node_u, target=node_v, weight='length')
                            if distance < min_distance:
                                min_distance = distance
                                closest_nodes = (node_u, node_v)
                        except nx.NetworkXNoPath:
                            continue
                if closest_nodes:
                    path = nx.shortest_path(graph, source=closest_nodes[0], target=closest_nodes[1], weight='length')
                    # Añadir aristas del camino a la ruta
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        if (u, v) not in route and (v, u) not in route:
                            routes[vehicle_id].append((u, v))
                subgraph.add_edges_from([(path[i], path[i + 1]) for i in range(len(path) - 1)])
                components = list(nx.connected_components(subgraph))

    return routes, vehicle_loads

def calculate_total_cost(routes, graph):
    """
    Calcula el costo total de todas las rutas.
    """
    total_cost = 0
    for route in routes:
        if not route:
            continue
        current_node = route[0][0]
        for u, v in route:
            # Añadir costo de moverse al siguiente arco
            try:
                path_length = nx.shortest_path_length(graph, source=current_node, target=u, weight='length')
                total_cost += path_length
            except nx.NetworkXNoPath:
                continue
            # Añadir costo de la arista
            edge_data = graph.get_edge_data(u, v)
            total_cost += edge_data[0]['length']
            current_node = v
    return total_cost

def shaking(routes, graph, k):
    """
    Genera una nueva solución en el vecindario 'k' mediante intercambios aleatorios.
    """
    new_routes = [route[:] for route in routes]

    for _ in range(k):
        # Seleccionar dos vehículos al azar con rutas no vacías
        vehicles_with_routes = [i for i in range(len(new_routes)) if new_routes[i]]
        if len(vehicles_with_routes) < 2:
            break  # No hay suficientes rutas para intercambiar

        v1, v2 = random.sample(vehicles_with_routes, 2)
        route1 = new_routes[v1]
        route2 = new_routes[v2]

        # Seleccionar arcos al azar para intercambiar
        idx1 = random.randint(0, len(route1) - 1)
        idx2 = random.randint(0, len(route2) - 1)

        # Intercambiar arcos
        route1[idx1], route2[idx2] = route2[idx2], route1[idx1]

        # Actualizar las rutas
        new_routes[v1] = route1
        new_routes[v2] = route2

    # Recalcular las cargas de los vehículos
    new_vehicle_loads = [
        sum(graph.get_edge_data(u, v)[0].get('demand', 1) for u, v in route)
        for route in new_routes
    ]

    return new_routes, new_vehicle_loads

def local_search(routes, graph, vehicle_capacity, vehicle_loads):
    """
    Realiza una búsqueda local para mejorar la solución actual.
    """
    improved = True
    best_routes = [route[:] for route in routes]
    best_vehicle_loads = vehicle_loads[:]
    best_cost = calculate_total_cost(best_routes, graph)

    while improved:
        improved = False
        for vehicle_id in range(len(best_routes)):
            route = best_routes[vehicle_id]
            for i in range(len(route)):
                for j in range(i + 1, len(route)):
                    # Intercambiar arcos y verificar factibilidad
                    new_route = route[:]
                    new_route[i], new_route[j] = new_route[j], new_route[i]

                    # Calcular la nueva carga
                    new_load = sum(
                        graph.get_edge_data(u, v)[0].get('demand', 1) for u, v in new_route
                    )

                    if new_load <= vehicle_capacity:
                        new_routes = best_routes[:]
                        new_routes[vehicle_id] = new_route
                        new_vehicle_loads = best_vehicle_loads[:]
                        new_vehicle_loads[vehicle_id] = new_load

                        new_cost = calculate_total_cost(new_routes, graph)
                        if new_cost < best_cost:
                            best_routes = new_routes
                            best_vehicle_loads = new_vehicle_loads
                            best_cost = new_cost
                            improved = True
        routes = best_routes
        vehicle_loads = best_vehicle_loads

    return best_routes, vehicle_loads

def variable_neighborhood_search(graph, num_vehicles, vehicle_capacity, max_k=3, max_iterations=50):
    """
    Implementa el algoritmo VNS para el PCARP.
    """
    # Inicialización con clustering para cubrir todas las aristas
    routes, vehicle_loads = initialize_routes_clustered(graph, num_vehicles, vehicle_capacity)
    best_routes = [route[:] for route in routes]
    best_vehicle_loads = vehicle_loads[:]
    best_cost = calculate_total_cost(best_routes, graph)

    iterations = 0
    while iterations < max_iterations:
        k = 1
        while k <= max_k:
            # Fase de Sacudida (Shaking)
            new_routes, new_vehicle_loads = shaking(routes, graph, k)
            # Fase de Búsqueda Local
            new_routes, new_vehicle_loads = local_search(new_routes, graph, vehicle_capacity, new_vehicle_loads)
            new_cost = calculate_total_cost(new_routes, graph)
            if new_cost < best_cost:
                best_routes = new_routes
                best_vehicle_loads = new_vehicle_loads
                best_cost = new_cost
                routes = new_routes  # Actualizar la solución actual
                vehicle_loads = new_vehicle_loads
                k = 1  # Reiniciar vecindad
                iterations = 0  # Reiniciar contador de iteraciones sin mejora
            else:
                k += 1
        iterations += 1

    return best_routes, best_cost

def plot_all_routes(graph, routes):
    """
    Dibuja todas las rutas generadas en un solo mapa con diferentes colores.
    """
    fig, ax = ox.plot_graph(graph, show=False, close=False, bgcolor='w')
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for vehicle_id, route in enumerate(routes):
        color = colors[vehicle_id % len(colors)]
        edge_lines = []
        for u, v in route:
            if graph.has_edge(u, v):
                edge_data = graph.get_edge_data(u, v)[0]
                x = [graph.nodes[u]['x'], graph.nodes[v]['x']]
                y = [graph.nodes[u]['y'], graph.nodes[v]['y']]
                ax.plot(x, y, color=color, linewidth=3, alpha=0.7)

    # Crear la carpeta "imagenes" si no existe
    if not os.path.exists('imagenes'):
        os.makedirs('imagenes')

    # Guardar la imagen con fecha y hora en el nombre
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f'VNS_all_vehicle_routes_{timestamp}.png'
    filepath = os.path.join('imagenes', filename)
    plt.title("Vehicle Routes")
    plt.savefig(filepath)
    plt.show()
    plt.close()

# -------------------------- EJECUCIÓN PRINCIPAL --------------------------

def main():
    """
    Ejecuta el algoritmo VNS para el PCARP y visualiza las rutas.
    """
    city_name = 'Maramburé, Luque, Paraguay'
    graph = ox.graph_from_place(city_name, network_type='drive')
    graph = ox.utils_graph.convert.to_undirected(graph)

    vehicle_capacity = 600
    num_vehicles = 3

    # Asignar demandas y longitudes a las aristas
    for u, v, key, data in graph.edges(keys=True, data=True):
        if 'length' not in data:
            data['length'] = random.uniform(50, 500)
        data['demand'] = random.randint(1, 10)

    # Ejecutar VNS
    best_routes, best_cost = variable_neighborhood_search(graph, num_vehicles, vehicle_capacity)

    # Verificar que todas las aristas con demanda fueron asignadas
    demand_edges = set((u, v) for u, v, data in graph.edges(data=True) if data.get('demand', 0) > 0)
    assigned_edges = set(edge for route in best_routes for edge in route)
    unassigned_edges = demand_edges - assigned_edges
    if unassigned_edges:
        print("Advertencia: Las siguientes aristas con demanda no fueron asignadas:")
        print(unassigned_edges)
    else:
        print("Todas las aristas con demanda fueron asignadas a las rutas.")

    print(f"Best total cost: {best_cost}")
    for i, route in enumerate(best_routes):
        print(f"Vehicle {i + 1} Route: {route}")

    # Visualizar rutas
    plot_all_routes(graph, best_routes)

if __name__ == "__main__":
    main()
