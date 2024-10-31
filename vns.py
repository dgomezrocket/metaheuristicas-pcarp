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
    Agrupa las aristas del grafo en clusters geográficos utilizando K-Means.

    Parámetros:
    - graph (networkx.Graph): Grafo de la ciudad obtenido mediante OSMnx.
    - num_clusters (int): Número de clusters o áreas en las que se desea dividir las aristas.

    Retorna:
    - edge_clusters (dict): Diccionario que asigna a cada arista (representada como un frozenset de nodos) un número de cluster.

    Descripción:
    Esta función calcula el punto medio de cada arista (es decir, el promedio de las coordenadas 'x' e 'y' de los nodos que la componen) y aplica el algoritmo K-Means para agruparlas en 'num_clusters' clusters basados en su posición geográfica.
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
        edge_key = frozenset({u, v})
        edge_clusters[edge_key] = labels[idx]

    return edge_clusters

def initialize_routes_clustered(graph, num_vehicles, vehicle_capacity):
    """
    Inicializa rutas asignando todas las aristas con demanda a los vehículos basándose en clusters geográficos.

    Parámetros:
    - graph (networkx.Graph): Grafo de la ciudad obtenido mediante OSMnx.
    - num_vehicles (int): Número de vehículos disponibles.
    - vehicle_capacity (int): Capacidad máxima de carga de cada vehículo.

    Retorna:
    - routes (list): Lista de rutas para cada vehículo; cada ruta es una lista de tuplas de aristas (u, v).
    - vehicle_loads (list): Lista de cargas actuales de cada vehículo.

    Descripción:
    La función agrupa las aristas con demanda en clusters y las asigna a los vehículos según el cluster al que pertenecen. Si un vehículo no tiene suficiente capacidad, se intenta asignar la arista a otro vehículo. Además, se asegura que las rutas de cada vehículo sean conexas utilizando un Árbol de Expansión Mínima (MST).

    Notas:
    - Las aristas se representan como frozensets para manejar grafos no dirigidos.
    - Se utiliza un MST para conectar los nodos dentro de cada ruta de manera eficiente.
    """
    routes = [[] for _ in range(num_vehicles)]
    vehicle_loads = [0] * num_vehicles

    # Agrupar aristas en clusters
    edge_clusters = cluster_edges(graph, num_vehicles)

    # Crear una lista de aristas con demanda
    demand_edges = [(u, v, data) for u, v, data in graph.edges(data=True) if data.get('demand', 0) > 0]

    # Asignar aristas a vehículos según el cluster
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

    # Asegurar que las rutas sean conectadas usando MST
    for vehicle_id in range(num_vehicles):
        route = routes[vehicle_id]
        if not route:
            continue
        # Construir subgrafo de la ruta
        subgraph = nx.Graph()
        nodes_in_route = set()
        for u, v in route:
            subgraph.add_edge(u, v)
            nodes_in_route.update([u, v])

        # Si el subgrafo no es conectado, construir MST
        if not nx.is_connected(subgraph):
            induced_subgraph = graph.subgraph(nodes_in_route)
            mst = nx.minimum_spanning_tree(induced_subgraph, weight='length')
            # Actualizar la ruta con las aristas del MST
            mst_edges = list(mst.edges())
            routes[vehicle_id].extend(mst_edges)
            subgraph = mst

    return routes, vehicle_loads

def calculate_total_cost(routes, graph):
    """
    Calcula el costo total de todas las rutas.

    Parámetros:
    - routes (list): Lista de rutas para cada vehículo; cada ruta es una lista de tuplas de aristas (u, v).
    - graph (networkx.Graph): Grafo de la ciudad obtenido mediante OSMnx.

    Retorna:
    - total_cost (float): Costo total acumulado de todas las rutas.

    Descripción:
    La función suma los costos de todas las aristas en las rutas, incluyendo el costo de moverse desde el nodo actual al inicio de la siguiente arista si es necesario. El costo se basa en la longitud de las aristas y los caminos más cortos entre nodos.

    Notas:
    - Utiliza el atributo 'length' de las aristas como medida de costo.
    - Si no hay un camino entre nodos, la excepción es manejada y se continúa.
    """
    total_cost = 0
    for route in routes:
        if not route:
            continue
        current_node = route[0][0]
        for u, v in route:
            # Añadir costo de moverse al siguiente arco si es necesario
            if current_node != u:
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

    Parámetros:
    - routes (list): Solución actual; lista de rutas para cada vehículo.
    - graph (networkx.Graph): Grafo de la ciudad obtenido mediante OSMnx.
    - k (int): Índice de vecindad que controla la intensidad del cambio (número de intercambios).

    Retorna:
    - new_routes (list): Nueva solución perturbada.
    - new_vehicle_loads (list): Nuevas cargas actuales de cada vehículo después de los intercambios.

    Descripción:
    La función perturba la solución actual realizando 'k' intercambios aleatorios de aristas entre rutas de vehículos distintos. Después de los intercambios, se recalculan las cargas de los vehículos para asegurar que no se exceda la capacidad.

    Notas:
    - Solo se realizan intercambios entre vehículos que tienen rutas no vacías.
    - Se verifica que haya suficientes vehículos con rutas para realizar los intercambios.
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
    Realiza una búsqueda local para mejorar la solución actual utilizando el movimiento 2-opt.

    Parámetros:
    - routes (list): Solución actual; lista de rutas para cada vehículo.
    - graph (networkx.Graph): Grafo de la ciudad obtenido mediante OSMnx.
    - vehicle_capacity (int): Capacidad máxima de carga de cada vehículo.
    - vehicle_loads (list): Cargas actuales de cada vehículo.

    Retorna:
    - best_routes (list): Mejor solución encontrada después de la búsqueda local.
    - vehicle_loads (list): Cargas actualizadas de cada vehículo.

    Descripción:
    La función busca mejorar la solución actual aplicando el movimiento 2-opt dentro de las rutas de cada vehículo. Intercambia segmentos de aristas para encontrar una ruta con menor costo, asegurando que no se exceda la capacidad del vehículo.

    Notas:
    - La búsqueda se detiene cuando no se encuentran más mejoras.
    - Se verifica que las nuevas rutas sean factibles en términos de capacidad.
    - Solo se considera el intercambio de segmentos adyacentes para eficiencia.
    """
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
                # Aplicar movimiento 2-opt
                new_route = route[:i] + route[i:i+2][::-1] + route[i+2:]
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
                        break  # Salir después de la mejora
            if improved:
                break  # Salir si se encontró una mejora
        routes = best_routes
        vehicle_loads = best_vehicle_loads

    return best_routes, vehicle_loads

def variable_neighborhood_search(graph, num_vehicles, vehicle_capacity, max_k=2, max_iterations=20):
    """
    Implementa el algoritmo de Búsqueda con Vecindad Variable (VNS) para el Problema de Enrutamiento de Arcos con Capacidad (PCARP).

    Parámetros:
    - graph (networkx.Graph): Grafo de la ciudad obtenido mediante OSMnx.
    - num_vehicles (int): Número de vehículos disponibles.
    - vehicle_capacity (int): Capacidad máxima de carga de cada vehículo.
    - max_k (int, opcional): Número máximo de vecindarios (profundidad de la sacudida). Valor por defecto es 2.
    - max_iterations (int, opcional): Número máximo de iteraciones sin mejora. Valor por defecto es 20.

    Retorna:
    - best_routes (list): Mejor conjunto de rutas encontradas.
    - best_cost (float): Costo total más bajo encontrado.

    Descripción:
    El algoritmo comienza con una solución inicial generada por clustering y luego explora diferentes vecindarios mediante la función de sacudida. Después de cada sacudida, aplica búsqueda local para intentar mejorar la solución. Si se encuentra una mejor solución, se actualiza y se reinicia la exploración de vecindarios.

    Notas:
    - Utiliza las funciones 'initialize_routes_clustered', 'shaking' y 'local_search'.
    - La exploración de vecindarios se realiza incrementando 'k' hasta 'max_k' o hasta encontrar una mejor solución.
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
    Dibuja todas las rutas generadas en un solo mapa con diferentes colores para cada vehículo.

    Parámetros:
    - graph (networkx.Graph): Grafo de la ciudad obtenido mediante OSMnx.
    - routes (list): Lista de rutas para cada vehículo; cada ruta es una lista de tuplas de aristas (u, v).

    Descripción:
    La función visualiza las rutas de todos los vehículos en un mapa, utilizando colores distintos para cada uno. Genera una imagen que se guarda en una carpeta 'imagenes' con un nombre que incluye la fecha y hora actual.

    Notas:
    - Si la carpeta 'imagenes' no existe, se crea automáticamente.
    - La imagen se muestra en pantalla y luego se cierra para liberar memoria.
    """
    fig, ax = ox.plot_graph(graph, show=False, close=False, bgcolor='w')
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for vehicle_id, route in enumerate(routes):
        color = colors[vehicle_id % len(colors)]
        for u, v in route:
            if graph.has_edge(u, v):
                x = [graph.nodes[u]['x'], graph.nodes[v]['x']]
                y = [graph.nodes[u]['y'], graph.nodes[v]['y']]
                ax.plot(x, y, color=color, linewidth=2, alpha=0.7)

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
    Función principal que ejecuta el algoritmo VNS para el PCARP y visualiza las rutas resultantes.

    Descripción:
    - Obtiene el grafo de una ciudad específica utilizando OSMnx.
    - Asigna demandas y longitudes aleatorias a las aristas del grafo.
    - Ejecuta el algoritmo de Búsqueda con Vecindad Variable (VNS) para encontrar las mejores rutas.
    - Verifica que todas las aristas con demanda fueron asignadas a alguna ruta.
    - Imprime el costo total y las rutas de cada vehículo.
    - Genera una visualización de las rutas y la guarda en una imagen.
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
    demand_edges = set(frozenset({u, v}) for u, v, data in graph.edges(data=True) if data.get('demand', 0) > 0)
    assigned_edges = set(frozenset({u, v}) for route in best_routes for u, v in route)
    unassigned_edges = demand_edges - assigned_edges
    if unassigned_edges:
        print("Advertencia: Las siguientes aristas con demanda no fueron asignadas:")
        for edge in unassigned_edges:
            print(f"Arista: {tuple(edge)}")
    else:
        print("Todas las aristas con demanda fueron asignadas a las rutas.")

    print(f"Best total cost: {best_cost}")
    for i, route in enumerate(best_routes):
        print(f"Vehicle {i + 1} Route: {route}")

    # Visualizar rutas
    plot_all_routes(graph, best_routes)

if __name__ == "__main__":
    main()
