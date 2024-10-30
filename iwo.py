import osmnx as ox
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import deque


def cluster_graph(graph, num_clusters):
    """
    Divide el grafo en áreas utilizando el algoritmo k-means clustering.
    """
    nodes = list(graph.nodes(data=True))
    coordinates = np.array([[data['y'], data['x']] for _, data in nodes])

    # Ejecutar k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(coordinates)
    labels = kmeans.labels_

    # Asignar etiquetas de clúster a los nodos
    for i, (node, data) in enumerate(nodes):
        data['cluster'] = labels[i]

    return labels, kmeans.cluster_centers_


def initialize_routes_by_area(graph, num_vehicles, vehicle_capacity, cluster_labels):
    """
    Inicializa rutas conectadas para los vehículos basándose en áreas específicas (clústeres).
    """
    routes = [[] for _ in range(num_vehicles)]
    vehicle_loads = [0] * num_vehicles

    # Asignar clusters a vehículos
    clusters = list(set(cluster_labels))
    cluster_vehicle_assignment = {cluster: i % num_vehicles for i, cluster in enumerate(clusters)}

    # Crear subgrafos por clúster
    cluster_subgraphs = {}
    for cluster in clusters:
        nodes_in_cluster = [node for node, data in graph.nodes(data=True) if data.get('cluster') == cluster]
        cluster_subgraphs[cluster] = graph.subgraph(nodes_in_cluster)

    # Construir rutas conectadas por vehículo
    for cluster, subgraph in cluster_subgraphs.items():
        vehicle_id = cluster_vehicle_assignment[cluster]
        # Obtener las aristas con demanda en el subgrafo
        demand_edges = [(u, v, data) for u, v, data in subgraph.edges(data=True) if data.get('demand', 0) > 0]
        if not demand_edges:
            continue

        # Crear un árbol de expansión mínima para asegurar conectividad
        mst = nx.minimum_spanning_tree(subgraph, weight='length')
        edges_in_mst = list(mst.edges(data=True))

        current_load = 0
        route = []
        for u, v, data in edges_in_mst:
            demand = data.get('demand', 1)
            if current_load + demand > vehicle_capacity:
                break
            route.append((u, v))
            current_load += demand

        routes[vehicle_id].extend(route)
        vehicle_loads[vehicle_id] = current_load

    return routes, vehicle_loads



def calculate_total_cost(routes, graph):
    """
    Calcula el costo total de todas las rutas, incluyendo desplazamientos entre aristas.
    """
    total_cost = 0
    for route in routes:
        if not route:
            continue
        # Iniciar en el primer nodo de la ruta
        current_node = route[0][0]
        for u, v in route:
            # Calcular el costo de moverse desde current_node a u
            if current_node != u:
                try:
                    path_length = nx.shortest_path_length(graph, source=current_node, target=u, weight='length')
                    total_cost += path_length
                except nx.NetworkXNoPath:
                    continue  # Si no hay camino, ignorar
            # Agregar el costo de la arista
            edge_data = graph.get_edge_data(u, v)
            total_cost += edge_data[0]['length']
            current_node = v
        # Regresar al depósito si es necesario (opcional)
    return total_cost



def mutate_solution_connected(solution, graph, vehicle_capacity):
    """
    Realiza una mutación que mantiene la conectividad de la ruta.
    """
    new_solution = [route[:] for route in solution]
    vehicle_id = random.choice([i for i in range(len(new_solution)) if new_solution[i]])

    route = new_solution[vehicle_id]
    if len(route) > 2:
        # Seleccionar un segmento de la ruta para invertir
        idx1, idx2 = sorted(random.sample(range(len(route)), 2))
        route[idx1:idx2] = reversed(route[idx1:idx2])

        # Verificar conectividad
        nodes_in_route = set()
        for u, v in route:
            nodes_in_route.update([u, v])
        subgraph = graph.subgraph(nodes_in_route)
        if not nx.is_connected(subgraph):
            # Si no es conectada, revertir la mutación
            new_solution[vehicle_id] = solution[vehicle_id][:]

        # Verificar capacidad
        total_demand = sum(graph.get_edge_data(u, v)[0].get('demand', 1) for u, v in route)
        if total_demand > vehicle_capacity:
            # Si excede la capacidad, revertir la mutación
            new_solution[vehicle_id] = solution[vehicle_id][:]

    return new_solution



def iwo_algorithm(graph, num_vehicles, vehicle_capacity, max_generations=100, initial_population=10, max_population=50):
    """
    Implementación del algoritmo de Invasive Weed Optimization (IWO) para PCARP con clustering de áreas.
    """
    num_clusters = num_vehicles
    cluster_labels, cluster_centers = cluster_graph(graph, num_clusters)

    # Inicializar la población de soluciones por áreas
    population = [initialize_routes_by_area(graph, num_vehicles, vehicle_capacity, cluster_labels)[0] for _ in
                  range(initial_population)]
    population_costs = [calculate_total_cost(route, graph) for route in population]
    best_routes = min(population, key=lambda x: calculate_total_cost(x, graph))
    best_cost = min(population_costs)

    # Ciclo principal de IWO
    for generation in range(max_generations):
        new_population = []

        for i in range(len(population)):
            num_offspring = int(
                (max_population - len(population)) * (1 - (population_costs[i] / max(population_costs))) + 1)
            for _ in range(num_offspring):
                new_solution = mutate_solution_connected(population[i], graph, vehicle_capacity)
                new_population.append(new_solution)

        new_population_costs = [calculate_total_cost(route, graph) for route in new_population]
        combined_population = population + new_population
        combined_costs = population_costs + new_population_costs

        sorted_indices = np.argsort(combined_costs)
        population = [combined_population[i] for i in sorted_indices[:max_population]]
        population_costs = [combined_costs[i] for i in sorted_indices[:max_population]]

        current_best_cost = min(population_costs)
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_routes = population[np.argmin(population_costs)]

    return best_routes, best_cost


def plot_all_routes(graph, routes):
    """
    Dibuja todas las rutas generadas en un solo mapa con diferentes colores para cada vehículo.
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
    plt.savefig('all_vehicle_routes.png')
    plt.show()
    plt.close()


def main_iwo():
    city_name = 'Maramburé, Luque, Paraguay'
    graph = ox.graph_from_place(city_name, network_type='drive')
    graph = ox.utils_graph.convert.to_undirected(graph)

    vehicle_capacity = 600
    num_vehicles = 3

    for u, v, key, data in graph.edges(keys=True, data=True):
        if 'length' not in data:
            data['length'] = random.uniform(50, 500)
        data['demand'] = random.randint(1, 10)

    best_routes, best_cost = iwo_algorithm(graph, num_vehicles, vehicle_capacity)
    print(f"Best total cost: {best_cost}")
    for i, route in enumerate(best_routes):
        print(f"Vehicle {i + 1} Route: {route}")

    plot_all_routes(graph, best_routes)


if __name__ == "__main__":
    main_iwo()
