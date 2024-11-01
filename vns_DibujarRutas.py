import osmnx as ox
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from sklearn.cluster import KMeans
import time
import matplotlib.animation as animation

# -------------------------- FUNCIONES PRINCIPALES DEL ALGORITMO VNS --------------------------

def cluster_edges(graph, num_clusters):
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
        edge_key = frozenset({u, v})
        edge_clusters[edge_key] = labels[idx]

    return edge_clusters

def initialize_routes_clustered(graph, num_vehicles, vehicle_capacity):
    routes = [[] for _ in range(num_vehicles)]
    vehicle_loads = [0] * num_vehicles

    edge_clusters = cluster_edges(graph, num_vehicles)

    demand_edges = [(u, v, data) for u, v, data in graph.edges(data=True) if data.get('demand', 0) > 0]

    for (u, v, data) in demand_edges:
        edge_key = frozenset({u, v})
        cluster = edge_clusters.get(edge_key)
        if cluster is None:
            continue
        vehicle_id = cluster % num_vehicles

        demand = data.get('demand', 1)
        if vehicle_loads[vehicle_id] + demand <= vehicle_capacity:
            routes[vehicle_id].append((u, v))
            vehicle_loads[vehicle_id] += demand
        else:
            assigned = False
            for vid in range(num_vehicles):
                if vehicle_loads[vid] + demand <= vehicle_capacity:
                    routes[vid].append((u, v))
                    vehicle_loads[vid] += demand
                    assigned = True
                    break
            if not assigned:
                print(f"No se pudo asignar la arista ({u}, {v}) debido a restricciones de capacidad.")

    for vehicle_id in range(num_vehicles):
        route = routes[vehicle_id]
        if not route:
            continue
        subgraph = nx.Graph()
        nodes_in_route = set()
        for u, v in route:
            subgraph.add_edge(u, v)
            nodes_in_route.update([u, v])

        if not nx.is_connected(subgraph):
            induced_subgraph = graph.subgraph(nodes_in_route)
            mst = nx.minimum_spanning_tree(induced_subgraph, weight='length')
            mst_edges = list(mst.edges())
            routes[vehicle_id].extend(mst_edges)
            subgraph = mst

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

def shaking(routes, graph, k):
    new_routes = [route[:] for route in routes]

    for _ in range(k):
        vehicles_with_routes = [i for i in range(len(new_routes)) if new_routes[i]]
        if len(vehicles_with_routes) < 2:
            break

        v1, v2 = random.sample(vehicles_with_routes, 2)
        route1 = new_routes[v1]
        route2 = new_routes[v2]

        idx1 = random.randint(0, len(route1) - 1)
        idx2 = random.randint(0, len(route2) - 1)

        route1[idx1], route2[idx2] = route2[idx2], route1[idx1]

        new_routes[v1] = route1
        new_routes[v2] = route2

    new_vehicle_loads = [
        sum(graph.get_edge_data(u, v)[0].get('demand', 1) for u, v in route)
        for route in new_routes
    ]

    return new_routes, new_vehicle_loads

def local_search(routes, graph, vehicle_capacity, vehicle_loads):
    improved = True
    best_routes = [route[:] for route in routes]
    best_vehicle_loads = vehicle_loads[:]
    best_cost = calculate_total_cost(best_routes, graph)

    while improved:
        improved = False
        for vehicle_id in range(len(best_routes)):
            route = best_routes[vehicle_id]
            route_length = len(route)
            if route_length < 2:
                continue
            for i in range(route_length - 1):
                new_route = route[:i] + route[i:i+2][::-1] + route[i+2:]
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
                        break
            if improved:
                break
        routes = best_routes
        vehicle_loads = best_vehicle_loads

    return best_routes, vehicle_loads

def variable_neighborhood_search(graph, num_vehicles, vehicle_capacity, max_k=2, max_iterations=20):
    routes, vehicle_loads = initialize_routes_clustered(graph, num_vehicles, vehicle_capacity)
    best_routes = [route[:] for route in routes]
    best_vehicle_loads = vehicle_loads[:]
    best_cost = calculate_total_cost(best_routes, graph)

    iterations = 0
    while iterations < max_iterations:
        k = 1
        while k <= max_k:
            new_routes, new_vehicle_loads = shaking(routes, graph, k)
            new_routes, new_vehicle_loads = local_search(new_routes, graph, vehicle_capacity, new_vehicle_loads)
            new_cost = calculate_total_cost(new_routes, graph)
            if new_cost < best_cost:
                best_routes = new_routes
                best_vehicle_loads = new_vehicle_loads
                best_cost = new_cost
                routes = new_routes
                vehicle_loads = new_vehicle_loads
                k = 1
                iterations = 0
            else:
                k += 1
        iterations += 1

    return best_routes, best_cost

# -------------------------- FUNCIONES DE VISUALIZACIÓN --------------------------

def plot_all_routes(graph, routes, algorithm_name="VNS"):
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

def animate_vehicle_routes(graph, vehicle_paths, algorithm_name="VNS"):
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

# -------------------------- EJECUCIÓN PRINCIPAL --------------------------

def main():
    start_time = time.time()

    city_name = 'Maramburé, Luque, Paraguay'
    graph = ox.graph_from_place(city_name, network_type='drive')
    graph = ox.utils_graph.convert.to_undirected(graph)

    vehicle_capacity = 600
    num_vehicles = 3

    for u, v, data in graph.edges(data=True):
        if 'length' not in data:
            data['length'] = random.uniform(50, 500)
        data['demand'] = random.randint(1, 10)

    best_routes, best_cost = variable_neighborhood_search(graph, num_vehicles, vehicle_capacity)

    print(f"Resultados del Algoritmo VNS:")
    for i, route in enumerate(best_routes):
        print(f"Ruta del vehículo {i + 1}: {route}")

    plot_all_routes(graph, best_routes, algorithm_name="VNS")
    animate_vehicle_routes(graph, get_vehicle_paths(best_routes, graph), algorithm_name="VNS")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos")

if __name__ == "__main__":
    main()
