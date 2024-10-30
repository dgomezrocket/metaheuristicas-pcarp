import osmnx as ox
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import deque

def cluster_graph(graph, num_clusters):
    """
    Divide el grafo en áreas utilizando el algoritmo de clustering k-means.

    Parámetros:
    - graph: Grafo de NetworkX que representa el mapa.
    - num_clusters: Número de clústeres en los que se dividirá el grafo.

    Retorna:
    - labels: Lista de etiquetas de clúster asignadas a cada nodo.
    - cluster_centers_: Coordenadas de los centros de los clústeres.

    Descripción:
    Esta función toma las coordenadas de los nodos del grafo y aplica k-means
    para agruparlos en clústeres. Luego, asigna a cada nodo la etiqueta del
    clúster al que pertenece.
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
    Inicializa rutas conectadas para los vehículos basándose en clústeres,
    asegurando que todas las aristas con demanda sean cubiertas.

    Parámetros:
    - graph: Grafo de NetworkX que representa el mapa.
    - num_vehicles: Número de vehículos disponibles.
    - vehicle_capacity: Capacidad máxima de carga de cada vehículo.
    - cluster_labels: Etiquetas de clúster asignadas a los nodos.

    Retorna:
    - routes: Lista de rutas para cada vehículo (listas de aristas).
    - vehicle_loads: Lista de cargas actuales de cada vehículo.

    Descripción:
    Esta función asigna todas las aristas con demanda a los vehículos,
    respetando la capacidad de cada uno. Las aristas se agrupan por clúster
    y se asignan inicialmente al vehículo correspondiente. Luego, se construyen
    rutas conectadas para cada vehículo, conectando componentes desconectados
    si es necesario.
    """
    routes = [[] for _ in range(num_vehicles)]
    vehicle_loads = [0] * num_vehicles

    # Obtener todas las aristas con demanda
    demand_edges = [(u, v, data) for u, v, data in graph.edges(data=True) if data.get('demand', 0) > 0]

    # Ordenar las aristas por clúster
    clusters = list(set(cluster_labels))
    edges_by_cluster = {cluster: [] for cluster in clusters}
    for u, v, data in demand_edges:
        cluster_u = graph.nodes[u].get('cluster')
        # Asignar la arista al clúster del nodo 'u'
        edges_by_cluster[cluster_u].append((u, v, data))

    # Asignar clústeres a vehículos
    cluster_vehicle_assignment = {cluster: i % num_vehicles for i, cluster in enumerate(clusters)}

    # Asignar aristas a vehículos respetando la capacidad
    for cluster in clusters:
        vehicle_id = cluster_vehicle_assignment[cluster]
        edges_in_cluster = edges_by_cluster[cluster]
        current_load = vehicle_loads[vehicle_id]

        for u, v, data in edges_in_cluster:
            demand = data.get('demand', 1)
            if current_load + demand > vehicle_capacity:
                # Buscar otro vehículo con capacidad disponible
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

    # Construir rutas conectadas para cada vehículo
    for vehicle_id in range(num_vehicles):
        route_edges = routes[vehicle_id]
        if not route_edges:
            continue
        # Crear un subgrafo con las aristas asignadas
        subgraph = nx.MultiGraph()
        for u, v in route_edges:
            edge_data = graph.get_edge_data(u, v)
            if edge_data:
                for key, data in edge_data.items():
                    subgraph.add_edge(u, v, key=key, **data)
            else:
                print(f"Arista ({u}, {v}) no encontrada en el grafo.")

        # Conectar el subgrafo usando caminos más cortos entre componentes
        if not nx.is_connected(subgraph.to_undirected()):
            components = list(nx.connected_components(subgraph.to_undirected()))
            while len(components) > 1:
                comp1 = components[0]
                comp2 = components[1]
                min_distance = float('inf')
                closest_pair = None
                for node_u in comp1:
                    for node_v in comp2:
                        try:
                            distance = nx.shortest_path_length(graph, source=node_u, target=node_v, weight='length')
                            if distance < min_distance:
                                min_distance = distance
                                closest_pair = (node_u, node_v)
                        except nx.NetworkXNoPath:
                            continue
                if closest_pair:
                    path = nx.shortest_path(graph, source=closest_pair[0], target=closest_pair[1], weight='length')
                    # Agregar las aristas del camino al subgrafo y a la ruta
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        edge_data = graph.get_edge_data(u, v)
                        if edge_data:
                            for key, data in edge_data.items():
                                subgraph.add_edge(u, v, key=key, **data)
                                routes[vehicle_id].append((u, v))
                components = list(nx.connected_components(subgraph.to_undirected()))

    return routes, vehicle_loads

def calculate_total_cost(routes, graph):
    """
    Calcula el costo total de todas las rutas, incluyendo desplazamientos entre aristas.

    Parámetros:
    - routes: Lista de rutas para cada vehículo (listas de aristas).
    - graph: Grafo de NetworkX que representa el mapa.

    Retorna:
    - total_cost: Costo total acumulado de todas las rutas.

    Descripción:
    Esta función suma el costo de recorrer cada arista en las rutas y el costo
    de desplazarse entre aristas no consecutivas (si es necesario), calculando
    el camino más corto entre ellas.
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
    Realiza una mutación que mantiene la conectividad de la ruta y la cobertura de todas las demandas.

    Parámetros:
    - solution: Solución actual (lista de rutas para cada vehículo).
    - graph: Grafo de NetworkX que representa el mapa.
    - vehicle_capacity: Capacidad máxima de carga de cada vehículo.

    Retorna:
    - new_solution: Nueva solución después de la mutación.

    Descripción:
    Esta función aplica una mutación a una ruta seleccionada al azar,
    invirtiendo un segmento de la ruta. Se asegura de que la mutación no
    rompa la conectividad de la ruta ni exceda la capacidad del vehículo.
    """
    new_solution = [route[:] for route in solution]
    vehicle_id = random.choice([i for i in range(len(new_solution)) if new_solution[i]])

    route = new_solution[vehicle_id]
    if len(route) > 2:
        # Seleccionar un segmento de la ruta para invertir
        idx1, idx2 = sorted(random.sample(range(len(route)), 2))
        mutated_route = route[:]
        mutated_route[idx1:idx2] = reversed(route[idx1:idx2])

        # Verificar conectividad
        nodes_in_route = set()
        for u, v in mutated_route:
            nodes_in_route.update([u, v])
        subgraph = graph.subgraph(nodes_in_route)
        if not nx.is_connected(subgraph):
            # Si no es conectada, no aplicar la mutación
            return solution

        # Verificar capacidad
        total_demand = sum(graph.get_edge_data(u, v)[0].get('demand', 1) for u, v in mutated_route)
        if total_demand > vehicle_capacity:
            # Si excede la capacidad, no aplicar la mutación
            return solution

        new_solution[vehicle_id] = mutated_route

    return new_solution

def iwo_algorithm(graph, num_vehicles, vehicle_capacity, max_generations=100, initial_population=10, max_population=50):
    """
    Implementación del algoritmo de Invasive Weed Optimization (IWO) para PCARP con clustering de áreas,
    asegurando la cobertura completa de las demandas.

    Parámetros:
    - graph: Grafo de NetworkX que representa el mapa.
    - num_vehicles: Número de vehículos disponibles.
    - vehicle_capacity: Capacidad máxima de carga de cada vehículo.
    - max_generations: Número máximo de generaciones para el algoritmo IWO.
    - initial_population: Tamaño de la población inicial.
    - max_population: Tamaño máximo de la población durante la evolución.

    Retorna:
    - best_routes: Las mejores rutas encontradas para cada vehículo.
    - best_cost: El costo total asociado a las mejores rutas.

    Descripción:
    Este algoritmo aplica IWO para encontrar rutas óptimas que cubran todas las
    aristas con demanda, respetando las capacidades de los vehículos y
    manteniendo las rutas conectadas. Utiliza mutaciones y selección basada en
    costos para evolucionar la población de soluciones.
    """
    num_clusters = num_vehicles
    cluster_labels, cluster_centers = cluster_graph(graph, num_clusters)

    # Obtener todas las aristas con demanda
    demand_edges = [(u, v) for u, v, data in graph.edges(data=True) if data.get('demand', 0) > 0]

    # Inicializar la población de soluciones por áreas
    population = []
    for _ in range(initial_population):
        routes, vehicle_loads = initialize_routes_by_area(graph, num_vehicles, vehicle_capacity, cluster_labels)
        # Verificar que se cubren todas las demandas
        assigned_edges = set(edge for route in routes for edge in route)
        if set(demand_edges).issubset(assigned_edges):
            population.append(routes)
        else:
            print("Advertencia: Inicialización no cubrió todas las demandas.")

    # Si no se pudo inicializar una población válida, abortar
    if not population:
        raise Exception("No se pudo inicializar una población válida que cubra todas las demandas.")

    population_costs = [calculate_total_cost(route, graph) for route in population]
    best_routes = min(population, key=lambda x: calculate_total_cost(x, graph))
    best_cost = min(population_costs)

    # Ciclo principal de IWO
    for generation in range(max_generations):
        new_population = []

        for i in range(len(population)):
            # Determinar el número de descendientes según el costo
            num_offspring = int(
                (max_population - len(population)) * (1 - (population_costs[i] / max(population_costs))) + 1)
            for _ in range(num_offspring):
                new_solution = mutate_solution_connected(population[i], graph, vehicle_capacity)
                # Verificar que se mantienen todas las demandas
                assigned_edges = set(edge for route in new_solution for edge in route)
                if set(demand_edges).issubset(assigned_edges):
                    new_population.append(new_solution)

        if not new_population:
            print("Advertencia: No se generaron nuevas soluciones válidas en la generación", generation)
            continue

        new_population_costs = [calculate_total_cost(route, graph) for route in new_population]
        combined_population = population + new_population
        combined_costs = population_costs + new_population_costs

        # Seleccionar las mejores soluciones para la siguiente generación
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

    Parámetros:
    - graph: Grafo de NetworkX que representa el mapa.
    - routes: Lista de rutas para cada vehículo (listas de aristas).

    Descripción:
    Esta función visualiza las rutas de todos los vehículos en el mapa, utilizando
    diferentes colores para distinguir cada ruta. Ayuda a visualizar la cobertura
    y conectividad de las rutas generadas.
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
    """
    Función principal que ejecuta el algoritmo IWO para resolver el problema CARP.

    Descripción:
    - Descarga y prepara el grafo de la ubicación especificada.
    - Asigna demandas aleatorias a las aristas.
    - Ejecuta el algoritmo IWO para encontrar las mejores rutas.
    - Verifica que todas las demandas han sido cubiertas.
    - Imprime el costo total y las rutas de cada vehículo.
    - Genera un gráfico visual de las rutas.
    """
    city_name = 'Maramburé, Luque, Paraguay'
    graph = ox.graph_from_place(city_name, network_type='drive')
    graph = ox.utils_graph.convert.to_undirected(graph)

    vehicle_capacity = 600
    num_vehicles = 3

    # Asignar demandas y longitudes aleatorias a las aristas (si es necesario)
    for u, v, key, data in graph.edges(keys=True, data=True):
        if 'length' not in data:
            data['length'] = random.uniform(50, 500)
        data['demand'] = random.randint(1, 10)

    best_routes, best_cost = iwo_algorithm(graph, num_vehicles, vehicle_capacity)

    # Verificar que todas las demandas están cubiertas
    demand_edges = set((u, v) for u, v, data in graph.edges(data=True) if data.get('demand', 0) > 0)
    assigned_edges = set(edge for route in best_routes for edge in route)
    uncovered_edges = demand_edges - assigned_edges
    if uncovered_edges:
        print("Las siguientes aristas con demanda no fueron asignadas:", uncovered_edges)
    else:
        print("Todas las aristas con demanda fueron asignadas.")

    print(f"Best total cost: {best_cost}")
    for i, route in enumerate(best_routes):
        print(f"Vehicle {i + 1} Route: {route}")

    plot_all_routes(graph, best_routes)

if __name__ == "__main__":
    main_iwo()
