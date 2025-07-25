import osmnx as ox
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    unvisited_edges = [(u, v, data) for u, v, data in graph.edges(data=True) if data.get('demand', 0) > 0]

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
                u, v, data = edge
                demand = data.get('demand', 1)
                if vehicle_loads[vehicle_id] + demand <= vehicle_capacity:
                    valid_edges.append(edge)

            # Si no hay arcos válidos, el vehículo se detiene
            if not valid_edges:
                break

            # Seleccionar el arco más cercano basado en la longitud
            nearest_edge = min(valid_edges, key=lambda edge: edge[2]['length'])
            unvisited_edges.remove(nearest_edge)

            # Actualizar la ruta y la carga del vehículo
            u, v, data = nearest_edge
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

        # Regresar al depósito (nodo inicial)
        if routes[vehicle_id]:
            routes[vehicle_id].append((current_node, start_node))

    return routes, vehicle_loads, unvisited_edges


def calculate_total_cost(routes, graph):
    """
    Calcula el costo total de todas las rutas generadas, incluyendo el costo de viajar entre aristas.

    Parámetros:
    - routes: Lista de rutas, cada una correspondiente a un vehículo.
    - graph: Grafo de la ciudad que contiene información de las aristas (longitud).

    Retorna:
    - total_cost: Costo total de todas las rutas en términos de longitud acumulada.
    """
    total_cost = 0
    for route in routes:
        if not route:
            continue
        # Suponiendo que el vehículo inicia y termina en un depósito (primer nodo de la ruta)
        depot = route[0][0]
        current_node = depot
        for edge in route:
            u, v = edge
            # Calcular el costo de ir desde current_node hasta u (inicio de la arista)
            if current_node != u:
                try:
                    length = nx.shortest_path_length(graph, source=current_node, target=u, weight='length')
                    total_cost += length
                except nx.NetworkXNoPath:
                    print(f"No hay camino entre {current_node} y {u}")
                    length = float('inf')
            # Añadir el costo de la arista (u, v)
            edge_data = graph.get_edge_data(u, v)
            if edge_data is None:
                edge_data = graph.get_edge_data(v, u)
                if edge_data is None:
                    print(f"No hay arista entre {u} y {v}")
                    continue
                else:
                    u, v = v, u  # Revertir para mantener el orden correcto
            # Obtener el primer key disponible
            edge_key = list(edge_data.keys())[0]
            edge_length = edge_data[edge_key]['length']
            total_cost += edge_length
            current_node = v  # Mover al siguiente nodo
        # Regresar al depósito
        if current_node != depot:
            try:
                length = nx.shortest_path_length(graph, source=current_node, target=depot, weight='length')
                total_cost += length
            except nx.NetworkXNoPath:
                print(f"No hay camino entre {current_node} y {depot}")
                length = float('inf')
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
        u, v, data = edge
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
    current_routes, vehicle_loads = assign_unvisited_edges(graph, current_routes, vehicle_loads, unvisited_edges,
                                                           vehicle_capacity)
    current_cost = calculate_total_cost(current_routes, graph)

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
            best_routes = current_routes[:]
            best_cost = current_cost

        temp *= cooling_rate

    return best_routes, best_cost


def generate_neighbor(routes, graph, vehicle_capacity, vehicle_loads):
    """
    Genera una solución vecina al intercambiar aristas entre rutas de diferentes vehículos.

    Parámetros:
    - routes: Rutas actuales de los vehículos.
    - graph: Grafo de la ciudad.
    - vehicle_capacity: Capacidad máxima de cada vehículo.
    - vehicle_loads: Carga acumulada de cada vehículo.

    Retorna:
    - new_routes: Nueva solución vecina generada por el intercambio de aristas.
    - new_vehicle_loads: Nueva carga acumulada de los vehículos.
    """
    new_routes = [route[:] for route in routes]
    new_vehicle_loads = vehicle_loads[:]

    # Seleccionar dos vehículos diferentes
    vehicle_ids = [i for i in range(len(new_routes)) if new_routes[i]]
    if len(vehicle_ids) < 2:
        return new_routes, new_vehicle_loads

    vehicle_id1, vehicle_id2 = random.sample(vehicle_ids, 2)

    route1 = new_routes[vehicle_id1]
    route2 = new_routes[vehicle_id2]

    # Seleccionar una arista aleatoriamente de cada ruta
    idx1 = random.randint(0, len(route1) - 1)
    idx2 = random.randint(0, len(route2) - 1)

    edge1 = route1[idx1]
    edge2 = route2[idx2]

    # Obtener la demanda para edge1
    edge_data1 = graph.get_edge_data(*edge1)
    if edge_data1 is None:
        edge_data1 = graph.get_edge_data(edge1[1], edge1[0])
        if edge_data1 is None:
            print(f"No hay arista entre {edge1[0]} y {edge1[1]}")
            return new_routes, new_vehicle_loads  # No se puede realizar el intercambio
        else:
            edge1 = (edge1[1], edge1[0])
    demand1 = edge_data1[list(edge_data1.keys())[0]].get('demand', 1)

    # Obtener la demanda para edge2
    edge_data2 = graph.get_edge_data(*edge2)
    if edge_data2 is None:
        edge_data2 = graph.get_edge_data(edge2[1], edge2[0])
        if edge_data2 is None:
            print(f"No hay arista entre {edge2[0]} y {edge2[1]}")
            return new_routes, new_vehicle_loads  # No se puede realizar el intercambio
        else:
            edge2 = (edge2[1], edge2[0])
    demand2 = edge_data2[list(edge_data2.keys())[0]].get('demand', 1)

    # Verificar si el intercambio es factible en términos de capacidad
    load1 = new_vehicle_loads[vehicle_id1] - demand1 + demand2
    load2 = new_vehicle_loads[vehicle_id2] - demand2 + demand1

    if load1 <= vehicle_capacity and load2 <= vehicle_capacity:
        # Realizar el intercambio
        new_routes[vehicle_id1][idx1], new_routes[vehicle_id2][idx2] = edge2, edge1
        new_vehicle_loads[vehicle_id1] = load1
        new_vehicle_loads[vehicle_id2] = load2

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
            else:
                print(f"No hay arista entre {u} y {v}")

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


def get_vehicle_paths(routes, graph):
    """
    Genera las rutas completas de los vehículos como listas de nodos,
    incluyendo el camino entre aristas utilizando el camino más corto.

    Parámetros:
    - routes: Lista de rutas (aristas) para cada vehículo.
    - graph: Grafo de la ciudad.

    Retorna:
    - vehicle_paths: Lista de rutas completas de nodos para cada vehículo.
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
            # Añadir el camino desde current_node hasta u si es necesario
            if current_node != u:
                try:
                    shortest_path = nx.shortest_path(graph, source=current_node, target=u, weight='length')
                    path.extend(shortest_path[1:])  # Excluir el current_node duplicado
                except nx.NetworkXNoPath:
                    print(f"No hay camino entre {current_node} y {u}")
                    continue
            # Añadir el camino de u a v (la arista actual)
            path.append(v)
            current_node = v
        # Regresar al depósito
        if current_node != route[0][0]:
            try:
                shortest_path = nx.shortest_path(graph, source=current_node, target=route[0][0], weight='length')
                path.extend(shortest_path[1:])
            except nx.NetworkXNoPath:
                print(f"No hay camino entre {current_node} y {route[0][0]}")
        vehicle_paths.append(path)
    return vehicle_paths


def animate_vehicle_routes(graph, vehicle_paths):
    """
    Crea una animación del recorrido de los vehículos en sus rutas.

    Parámetros:
    - graph: Grafo de la ciudad.
    - vehicle_paths: Lista de rutas completas de nodos para cada vehículo.
    """
    fig, ax = ox.plot_graph(graph, show=False, close=False, bgcolor='w')

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    max_len = max(len(path) for path in vehicle_paths)
    lines = []
    texts = []
    for i, path in enumerate(vehicle_paths):
        color = colors[i % len(colors)]
        line, = ax.plot([], [], color=color, linewidth=3, alpha=0.7)
        lines.append(line)
        text = ax.text(0.02, 0.95 - i * 0.05, '', transform=ax.transAxes, color=color)
        texts.append(text)

    def init():
        for line in lines:
            line.set_data([], [])
        for text in texts:
            text.set_text('')
        return lines + texts

    def animate(frame):
        for i, path in enumerate(vehicle_paths):
            if frame < len(path) - 1:
                x = [graph.nodes[n]['x'] for n in path[:frame + 1]]
                y = [graph.nodes[n]['y'] for n in path[:frame + 1]]
                lines[i].set_data(x, y)
                # Calcular la distancia acumulada
                distance = 0
                for j in range(frame):
                    u = path[j]
                    v = path[j + 1]
                    try:
                        edge_data = graph.get_edge_data(u, v)
                        if edge_data is None:
                            edge_data = graph.get_edge_data(v, u)
                        edge_key = list(edge_data.keys())[0]
                        edge_length = edge_data[edge_key]['length']
                        distance += edge_length
                    except:
                        continue
                texts[i].set_text(f'Vehículo {i + 1}: {distance:.2f} m')
        return lines + texts

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=max_len, interval=500, blit=True)

    # Guardar la animación como GIF
    anim.save('vehicle_routes_animation.gif', writer='pillow', fps=2)
    plt.show()

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
    graph = ox.utils_graph.convert.to_undirected(graph)  # Reemplazar la función deprecada

    vehicle_capacity = 600  # Capacidad de cada vehículo
    num_vehicles = 3  # Número de vehículos

    # Inicializar el grafo con longitud y demanda
    edge_count = 0
    total_demand_edges = 0
    for u, v, data in graph.edges(data=True):
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

    # Obtener las rutas completas de los vehículos
    vehicle_paths = get_vehicle_paths(best_routes, graph)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nResultados del Algoritmo de Simulated Annealing:")
    print(f"Costo total de las rutas: {best_cost}")
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos")

    for i, route in enumerate(best_routes):
        print(f"Ruta del Vehículo {i + 1}: {route}")

    # Verificar si todas las aristas con demanda fueron asignadas
    demand_edges = set(frozenset({u, v}) for u, v, data in graph.edges(data=True) if data.get('demand', 0) > 0)
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

    # Crear la animación de los vehículos
    animate_vehicle_routes(graph, vehicle_paths)


if __name__ == "__main__":
    main()
