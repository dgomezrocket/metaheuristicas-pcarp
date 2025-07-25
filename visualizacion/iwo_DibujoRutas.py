import osmnx as ox
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import os
import matplotlib.animation as animation
from sklearn.cluster import KMeans

# -------------------------- FUNCIONES DE VISUALIZACIÓN --------------------------

def plot_all_routes(graph, routes, algorithm_name="Routes"):
    """
    Dibuja todas las rutas generadas en un solo mapa con diferentes colores para cada vehículo.

    Parámetros:
    - graph: Grafo de la ciudad.
    - routes: Lista de rutas para cada vehículo.
    - algorithm_name: Nombre del algoritmo (para el título y archivo).
    """
    vehicle_paths = get_vehicle_paths(routes, graph)
    fig, ax = ox.plot_graph(graph, show=False, close=False, bgcolor='w')

    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for vehicle_id, path in enumerate(vehicle_paths):
        color = colors[vehicle_id % len(colors)]
        x = [graph.nodes[n]['x'] for n in path]
        y = [graph.nodes[n]['y'] for n in path]
        ax.plot(x, y, color=color, linewidth=3, alpha=0.7)

    if not os.path.exists('imagenes'):
        os.makedirs('imagenes')

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    plt.title(f"{algorithm_name} - Vehicle Routes")
    filename = f'{algorithm_name}_all_vehicle_routes_{timestamp}.png'
    filepath = os.path.join('imagenes', filename)
    plt.savefig(filepath)
    plt.show()
    plt.close()

def get_vehicle_paths(routes, graph):
    """
    Genera las rutas completas de los vehículos como listas de nodos, incluyendo el camino entre aristas.
    """
    vehicle_paths = []
    for route in routes:
        if not route:
            vehicle_paths.append([])
            continue
        path = []
        current_node = route[0][0]
        path.append(current_node)
        for edge in route:
            u, v = edge
            if current_node != u:
                try:
                    shortest_path = nx.shortest_path(graph, source=current_node, target=u, weight='length')
                    path.extend(shortest_path[1:])
                except nx.NetworkXNoPath:
                    print(f"No hay camino entre {current_node} y {u}")
                    continue
            try:
                shortest_path = nx.shortest_path(graph, source=u, target=v, weight='length')
                path.extend(shortest_path[1:])
            except nx.NetworkXNoPath:
                print(f"No hay camino entre {u} y {v}")
                continue
            current_node = v
        vehicle_paths.append(path)
    return vehicle_paths

def animate_vehicle_routes(graph, vehicle_paths, algorithm_name="Routes"):
    """
    Crea una animación del recorrido de los vehículos en sus rutas.
    """
    fig, ax = ox.plot_graph(graph, show=False, close=False, bgcolor='w')

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    max_len = max(len(path) for path in vehicle_paths)

    lines = []
    for i in range(len(vehicle_paths)):
        color = colors[i % len(colors)]
        line, = ax.plot([], [], color=color, linewidth=2)
        lines.append(line)

    points = []
    for i in range(len(vehicle_paths)):
        color = colors[i % len(colors)]
        point, = ax.plot([], [], marker='o', color=color, markersize=8)
        points.append(point)

    def init():
        for line in lines:
            line.set_data([], [])
        for point in points:
            point.set_data([], [])
        return lines + points

    def animate(frame):
        for i, path in enumerate(vehicle_paths):
            if frame < len(path):
                x = [graph.nodes[n]['x'] for n in path[:frame + 1]]
                y = [graph.nodes[n]['y'] for n in path[:frame + 1]]
                lines[i].set_data(x, y)
                vehicle_x = graph.nodes[path[frame]]['x']
                vehicle_y = graph.nodes[path[frame]]['y']
                points[i].set_data([vehicle_x], [vehicle_y])
            else:
                x = [graph.nodes[n]['x'] for n in path]
                y = [graph.nodes[n]['y'] for n in path]
                lines[i].set_data(x, y)
                points[i].set_data([], [])
        return lines + points

    ani = animation.FuncAnimation(fig, animate, frames=max_len, init_func=init,
                                  interval=200, blit=True, repeat=False)
    gif_filename = f'{algorithm_name}_vehicle_routes_animation.gif'
    ani.save(gif_filename, writer='pillow')
    plt.show()
    plt.close()

# -------------------------- FUNCIONES PRINCIPALES DEL ALGORITMO IWO --------------------------

def cluster_graph(graph, num_clusters):
    nodes = list(graph.nodes(data=True))
    coordinates = np.array([[data['y'], data['x']] for _, data in nodes])

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(coordinates)
    labels = kmeans.labels_

    for i, (node, data) in enumerate(nodes):
        data['cluster'] = labels[i]

    return labels, kmeans.cluster_centers_

def initialize_routes_by_area(graph, num_vehicles, vehicle_capacity, cluster_labels):
    routes = [[] for _ in range(num_vehicles)]
    vehicle_loads = [0] * num_vehicles

    demand_edges = [(u, v, data) for u, v, data in graph.edges(data=True) if data.get('demand', 0) > 0]

    clusters = list(set(cluster_labels))
    edges_by_cluster = {cluster: [] for cluster in clusters}
    for u, v, data in demand_edges:
        cluster_u = graph.nodes[u].get('cluster')
        edges_by_cluster[cluster_u].append((u, v, data))

    cluster_vehicle_assignment = {cluster: i % num_vehicles for i, cluster in enumerate(clusters)}

    for cluster in clusters:
        vehicle_id = cluster_vehicle_assignment[cluster]
        edges_in_cluster = edges_by_cluster[cluster]
        current_load = vehicle_loads[vehicle_id]

        for u, v, data in edges_in_cluster:
            demand = data.get('demand', 1)
            if current_load + demand > vehicle_capacity:
                assigned = False
                for vid in range(num_vehicles):
                    if vehicle_loads[vid] + demand <= vehicle_capacity:
                        routes[vid].append((u, v))
                        vehicle_loads[vid] += demand
                        assigned = True
                        break
                if not assigned:
                    print("Advertencia: Demanda excede la capacidad total disponible.")
            else:
                routes[vehicle_id].append((u, v))
                vehicle_loads[vehicle_id] += demand

    return routes, vehicle_loads

def calculate_total_cost(routes, graph):
    total_cost = 0
    for route in routes:
        if not route:
            continue
        current_node = route[0][0]
        for u, v in route:
            if current_node != u:
                try:
                    path_length = nx.shortest_path_length(graph, source=current_node, target=u, weight='length')
                    total_cost += path_length
                except nx.NetworkXNoPath:
                    continue
            edge_data = graph.get_edge_data(u, v)
            total_cost += edge_data[0]['length']
            current_node = v
    return total_cost

def mutate_solution_connected(solution, graph, vehicle_capacity):
    new_solution = [route[:] for route in solution]
    vehicle_id = random.choice([i for i in range(len(new_solution)) if new_solution[i]])

    route = new_solution[vehicle_id]
    if len(route) > 2:
        idx1, idx2 = sorted(random.sample(range(len(route)), 2))
        mutated_route = route[:]
        mutated_route[idx1:idx2] = reversed(route[idx1:idx2])

        nodes_in_route = set()
        for u, v in mutated_route:
            nodes_in_route.update([u, v])
        subgraph = graph.subgraph(nodes_in_route)
        if not nx.is_connected(subgraph):
            return solution

        total_demand = sum(graph.get_edge_data(u, v)[0].get('demand', 1) for u, v in mutated_route)
        if total_demand > vehicle_capacity:
            return solution

        new_solution[vehicle_id] = mutated_route

    return new_solution

def iwo_algorithm(graph, num_vehicles, vehicle_capacity, max_generations=100, initial_population=10, max_population=50):
    num_clusters = num_vehicles
    cluster_labels, cluster_centers = cluster_graph(graph, num_clusters)

    demand_edges = [(u, v) for u, v, data in graph.edges(data=True) if data.get('demand', 0) > 0]

    population = []
    for _ in range(initial_population):
        routes, vehicle_loads = initialize_routes_by_area(graph, num_vehicles, vehicle_capacity, cluster_labels)
        assigned_edges = set(edge for route in routes for edge in route)
        if set(demand_edges).issubset(assigned_edges):
            population.append(routes)
        else:
            print("Advertencia: Inicialización no cubrió todas las demandas.")

    if not population:
        raise Exception("No se pudo inicializar una población válida que cubra todas las demandas.")

    population_costs = [calculate_total_cost(route, graph) for route in population]
    best_routes = min(population, key=lambda x: calculate_total_cost(x, graph))
    best_cost = min(population_costs)

    for generation in range(max_generations):
        new_population = []

        for i in range(len(population)):
            num_offspring = int((max_population - len(population)) * (1 - (population_costs[i] / max(population_costs))) + 1)
            for _ in range(num_offspring):
                new_solution = mutate_solution_connected(population[i], graph, vehicle_capacity)
                assigned_edges = set(edge for route in new_solution for edge in route)
                if set(demand_edges).issubset(assigned_edges):
                    new_population.append(new_solution)

        if not new_population:
            print("Advertencia: No se generaron nuevas soluciones válidas en la generación", generation)
            continue

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

# -------------------------- EJECUCIÓN PRINCIPAL --------------------------

def main_iwo():
    city_name = 'Maramburé, Luque, Paraguay'
    graph = ox.graph_from_place(city_name, network_type='drive')
    graph = ox.utils_graph.convert.to_undirected(graph)

    vehicle_capacity = 600
    num_vehicles = 3

    for u, v, data in graph.edges(data=True):
        if 'length' not in data:
            data['length'] = random.uniform(50, 500)
        data['demand'] = random.randint(1, 10)

    best_routes_iwo, best_cost_iwo = iwo_algorithm(graph, num_vehicles, vehicle_capacity)
    print("Resultados de Invasive Weed Optimization:")
    for i, route in enumerate(best_routes_iwo):
        print(f"Ruta del vehículo {i + 1}: {route}")

    plot_all_routes(graph, best_routes_iwo, algorithm_name="IWO")
    animate_vehicle_routes(graph, get_vehicle_paths(best_routes_iwo, graph), algorithm_name="IWO")

if __name__ == "__main__":
    main_iwo()
