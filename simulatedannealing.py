import osmnx as ox
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import time

# -------------------------- FUNCIONES PRINCIPALES --------------------------

def initialize_routes_nearest_neighbor(graph, num_vehicles, vehicle_capacity):
    """
    Inicializa rutas para los vehículos utilizando el algoritmo de Vecino Más Cercano.

    Parámetros:
    - graph: Grafo de la ciudad con arcos que representan las calles y nodos que representan las intersecciones.
    - num_vehicles: Número de vehículos disponibles para la ruta.
    - vehicle_capacity: Capacidad máxima de cada vehículo para recolectar demanda.

    Retorna:
    - routes: Lista de rutas, cada una correspondiente a un vehículo.
    - vehicle_loads: Lista de la carga acumulada de cada vehículo.
    - unvisited_edges: Lista de arcos que aún no han sido visitados por los vehículos.
    """
    # Inicialización de variables
    routes = [[] for _ in range(num_vehicles)]
    vehicle_loads = [0] * num_vehicles
    unvisited_edges = list(graph.edges(keys=True, data=True))

    # Seleccionar nodos de inicio aleatorios para cada vehículo
    start_nodes = random.sample(list(graph.nodes), num_vehicles)

    # Construir rutas para cada vehículo usando el vecino más cercano
    for vehicle_id, start_node in enumerate(start_nodes):
        current_node = start_node
        while unvisited_edges:
            # Filtrar arcos que conectan con el nodo actual y cumplen con la capacidad del vehículo
            edges_from_node = [edge for edge in unvisited_edges if edge[0] == current_node or edge[1] == current_node]
            valid_edges = []
            for edge in edges_from_node:
                u, v, key, data = edge
                demand = data.get('demand', 1)
                if vehicle_loads[vehicle_id] + demand <= vehicle_capacity:
                    valid_edges.append(edge)

            # Si no hay arcos válidos, el vehículo se detiene
            if not valid_edges:
                break

            # Seleccionar el arco más cercano basado en la longitud
            nearest_edge = min(valid_edges, key=lambda edge: edge[3]['length'])
            unvisited_edges.remove(nearest_edge)

            # Actualizar la ruta y la carga del vehículo
            u, v, key, data = nearest_edge
            vehicle_loads[vehicle_id] += data['demand']

            if not routes[vehicle_id]:
                routes[vehicle_id].append((u, v))
                current_node = v if current_node == u else u
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
    """
    Calcula el costo total de todas las rutas, incluyendo el costo de recorrer las aristas en las rutas
    y el costo de desplazamiento entre las aristas (costo muerto o "deadhead cost").

    Parámetros:
    - routes: Lista de rutas, donde cada ruta es una lista de aristas representadas como tuplas (u, v).
      Cada ruta corresponde al camino que sigue un vehículo.
    - graph: Grafo de NetworkX que representa el mapa de la ciudad, con atributos en las aristas que incluyen 'length'.

    Retorna:
    - total_cost: Costo total acumulado de todas las rutas, calculado como la suma de las longitudes de todas
      las aristas recorridas, incluyendo tanto las aristas de servicio como los caminos tomados para moverse entre ellas.
    """
    total_cost = 0
    for route in routes:
        if not route:
            continue  # Saltar rutas vacías
        # Iniciar en el primer nodo de la primera arista en la ruta
        current_node = route[0][0]
        for u, v in route:
            # Calcular el costo de moverse desde current_node al nodo inicial u de la arista actual
            if current_node != u:
                try:
                    # Encontrar la longitud del camino más corto entre current_node y u
                    path_length = nx.shortest_path_length(graph, source=current_node, target=u, weight='length')
                    total_cost += path_length
                except nx.NetworkXNoPath:
                    # Si no hay camino, imprimir una advertencia y continuar
                    print(f"No hay camino entre {current_node} y {u}.")
                    continue
            # Agregar el costo de recorrer la arista actual (u, v)
            if graph.has_edge(u, v):
                edge_data = graph.get_edge_data(u, v)
                # Si existen múltiples aristas entre u y v, seleccionar la primera
                total_cost += edge_data[0]['length']
            elif graph.has_edge(v, u):
                edge_data = graph.get_edge_data(v, u)
                total_cost += edge_data[0]['length']
            else:
                # Si la arista no existe en el grafo, imprimir una advertencia
                print(f"Arista ({u}, {v}) no encontrada en el grafo.")
            # Actualizar current_node al nodo final v de la arista actual
            current_node = v
    return total_cost



def assign_unvisited_edges(graph, routes, vehicle_loads, unvisited_edges, vehicle_capacity):
    """
    Asigna arcos no visitados a los vehículos, priorizando la capacidad y distancia mínima.

    Parámetros:
    - graph: Grafo de la ciudad.
    - routes: Lista de rutas actuales para cada vehículo.
    - vehicle_loads: Lista de carga acumulada de cada vehículo.
    - unvisited_edges: Lista de arcos no visitados.
    - vehicle_capacity: Capacidad máxima de cada vehículo.

    Retorna:
    - routes: Rutas actualizadas para cada vehículo.
    - vehicle_loads: Carga actualizada de cada vehículo.
    """
    for edge in unvisited_edges:
        u, v, key, data = edge
        demand = data['demand']

        min_distance = float('inf')
        best_vehicle = None
        best_insert_pos = None

        # Buscar el vehículo más cercano con capacidad suficiente
        for vehicle_id, route in enumerate(routes):
            if vehicle_loads[vehicle_id] + demand <= vehicle_capacity:
                for idx, (node_u, node_v) in enumerate(route):
                    try:
                        distance_to_u = nx.shortest_path_length(graph, source=node_u, target=u, weight='length')
                    except nx.NetworkXNoPath:
                        distance_to_u = float('inf')
                    try:
                        distance_to_v = nx.shortest_path_length(graph, source=node_v, target=v, weight='length')
                    except nx.NetworkXNoPath:
                        distance_to_v = float('inf')

                    # Actualizar la posición más cercana para insertar el arco
                    if distance_to_u < min_distance:
                        min_distance = distance_to_u
                        best_vehicle = vehicle_id
                        best_insert_pos = idx

                    if distance_to_v < min_distance:
                        min_distance = distance_to_v
                        best_vehicle = vehicle_id
                        best_insert_pos = idx

        # Asignar el arco al mejor vehículo encontrado
        if best_vehicle is not None:
            routes[best_vehicle].insert(best_insert_pos + 1, (u, v))
            vehicle_loads[best_vehicle] += demand

    return routes, vehicle_loads


def simulated_annealing(graph, num_vehicles, vehicle_capacity, initial_temp=1200, cooling_rate=0.95, min_temp=1):
    """
    Implementa el algoritmo de Simulated Annealing para PCARP.

    Parámetros:
    - graph: Grafo de la ciudad.
    - num_vehicles: Número de vehículos disponibles para el enrutamiento.
    - vehicle_capacity: Capacidad máxima de cada vehículo.
    - initial_temp: Temperatura inicial para el algoritmo.
    - cooling_rate: Tasa de enfriamiento de la temperatura.
    - min_temp: Temperatura mínima para detener el algoritmo.

    Retorna:
    - best_routes: Mejor conjunto de rutas encontradas.
    - best_cost: Costo total más bajo encontrado.
    """
    current_routes, vehicle_loads, unvisited_edges = initialize_routes_nearest_neighbor(graph, num_vehicles,
                                                                                        vehicle_capacity)
    current_cost = calculate_total_cost(current_routes, graph)

    current_routes, vehicle_loads = assign_unvisited_edges(graph, current_routes, vehicle_loads, unvisited_edges,
                                                           vehicle_capacity)
    best_routes = current_routes[:]
    best_cost = current_cost

    temp = initial_temp

    # Iterar mientras la temperatura sea mayor a la mínima
    while temp > min_temp:
        new_routes, new_vehicle_loads = generate_neighbor(current_routes, graph, vehicle_capacity, vehicle_loads)
        new_cost = calculate_total_cost(new_routes, graph)

        # Aceptar la nueva solución si mejora el costo o por probabilidad
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
    """
    Genera una solución vecina al intercambiar dos arcos dentro de una ruta.

    Parámetros:
    - routes: Rutas actuales de los vehículos.
    - graph: Grafo de la ciudad.
    - vehicle_capacity: Capacidad máxima de cada vehículo.
    - vehicle_loads: Carga acumulada de cada vehículo.

    Retorna:
    - new_routes: Nueva solución vecina generada por el intercambio de arcos.
    - new_vehicle_loads: Nueva carga acumulada de los vehículos.
    """
    new_routes = [route[:] for route in routes]
    new_vehicle_loads = vehicle_loads[:]

    # Seleccionar un vehículo con al menos una ruta y realizar el intercambio
    vehicle_id = random.choice([i for i in range(len(new_routes)) if new_routes[i]])
    if len(new_routes[vehicle_id]) > 1:
        idx1, idx2 = random.sample(range(len(new_routes[vehicle_id])), 2)
        new_routes[vehicle_id][idx1], new_routes[vehicle_id][idx2] = new_routes[vehicle_id][idx2], \
        new_routes[vehicle_id][idx1]

    return new_routes, new_vehicle_loads


def plot_all_routes(graph, routes):
    """
    Dibuja todas las rutas generadas en un solo mapa con diferentes colores para cada vehículo.

    Parámetros:
    - graph: Grafo de la ciudad.
    - routes: Rutas de los vehículos a visualizar.
    """
    fig, ax = ox.plot_graph(graph, show=False, close=False, bgcolor='w')

    # Colores para cada vehículo
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for vehicle_id, route in enumerate(routes):
        color = colors[vehicle_id % len(colors)]
        for u, v in route:
            if graph.has_edge(u, v):
                x = [graph.nodes[u]['x'], graph.nodes[v]['x']]
                y = [graph.nodes[u]['y'], graph.nodes[v]['y']]
                ax.plot(x, y, color=color, linewidth=3, alpha=0.7)
            elif graph.has_edge(v, u):  # Considerar aristas en sentido contrario
                x = [graph.nodes[v]['x'], graph.nodes[u]['x']]
                y = [graph.nodes[v]['y'], graph.nodes[u]['y']]
                ax.plot(x, y, color=color, linewidth=3, alpha=0.7)

    # Crear la carpeta "imagenes" si no existe
    if not os.path.exists('imagenes'):
        os.makedirs('imagenes')

    # Obtener fecha y hora actual
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Guardar la imagen con fecha y hora en el nombre
    plt.title("Vehicle Routes")
    filename = f'SA_all_vehicle_routes_{timestamp}.png'
    filepath = os.path.join('imagenes', filename)
    plt.savefig(filepath)
    plt.show()
    plt.close()


# -------------------------- EJECUCIÓN PRINCIPAL --------------------------

def main():
    """
    Ejecuta el algoritmo de PCARP utilizando Simulated Annealing en un grafo de la ciudad.
    Genera las mejores rutas para los vehículos y las visualiza en un mapa.
    """
    # Medir el tiempo de ejecución
    start_time = time.time()

    city_name = 'Maramburé, Luque, Paraguay'
    graph = ox.graph_from_place(city_name, network_type='drive')
    graph = ox.utils_graph.convert.to_undirected(graph)

    vehicle_capacity = 600  # Capacidad de cada vehículo
    num_vehicles = 3  # Número de vehículos

    # Inicializar el grafo con longitud y demanda
    edge_count = 0
    total_demand_edges = 0
    for u, v, key, data in graph.edges(keys=True, data=True):
        edge_count += 1
        if 'length' not in data:
            data['length'] = random.uniform(50, 500)
        data['demand'] = random.randint(1, 10)
        if data['demand'] > 0:
            total_demand_edges += 1

    node_count = graph.number_of_nodes()

    print(f"Información del Grafo:")
    print(f"Número de nodos: {node_count}")
    print(f"Número de aristas: {edge_count}")
    print(f"Número de aristas con demanda: {total_demand_edges}")



    # Ejecutar Simulated Annealing para encontrar las mejores rutas
    best_routes, best_cost = simulated_annealing(graph, num_vehicles, vehicle_capacity)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nResultados del Algoritmo de Simulated Annealing:")
    print(f"Costo total de las rutas: {best_cost}")
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos")

    for i, route in enumerate(best_routes):
        print(f"Ruta del Vehículo {i + 1}: {route}")

    # Verificar si todas las aristas con demanda fueron asignadas
    demand_edges = set(frozenset({u, v}) for u, v, key, data in graph.edges(keys=True, data=True) if data.get('demand', 0) > 0)
    assigned_edges = set(frozenset({u, v}) for route in best_routes for u, v in route)
    unassigned_edges = demand_edges - assigned_edges
    if unassigned_edges:
        print("\nAdvertencia: Las siguientes aristas con demanda no fueron asignadas:")
        for edge in unassigned_edges:
            print(f"Arista: {tuple(edge)}")
    else:
        print("\nTodas las aristas con demanda fueron asignadas a las rutas.")

    # Generar una sola imagen con todas las rutas
    plot_all_routes(graph, best_routes)


if __name__ == "__main__":
    main()
