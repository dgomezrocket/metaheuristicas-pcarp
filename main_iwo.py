import osmnx as ox
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt


# Configuración de IWO
def iwo_algorithm(graph, num_vehicles, vehicle_capacity, max_generations=100, initial_population=10, max_population=50):
    """
    Implementación del algoritmo de Invasive Weed Optimization (IWO) para CARP.

    Parámetros:
    - graph: Grafo de la ciudad.
    - num_vehicles: Número de vehículos disponibles para la ruta.
    - vehicle_capacity: Capacidad máxima de cada vehículo.
    - max_generations: Número máximo de generaciones.
    - initial_population: Tamaño de la población inicial.
    - max_population: Tamaño máximo de la población.

    Retorna:
    - best_routes: Mejor conjunto de rutas encontradas.
    - best_cost: Costo total más bajo encontrado.
    """
    # Inicializar la población de soluciones
    population = [initialize_routes_nearest_neighbor(graph, num_vehicles, vehicle_capacity)[0] for _ in
                  range(initial_population)]
    population_costs = [calculate_total_cost(route, graph) for route in population]
    best_routes = min(population, key=lambda x: calculate_total_cost(x, graph))
    best_cost = min(population_costs)

    # Ciclo principal de IWO
    for generation in range(max_generations):
        new_population = []

        # Reproducción de soluciones
        for i in range(len(population)):
            num_offspring = int(
                (max_population - len(population)) * (1 - (population_costs[i] / max(population_costs))) + 1)

            for _ in range(num_offspring):
                # Mutación y generación de nuevos individuos
                new_solution = mutate_solution(population[i], graph, vehicle_capacity)
                new_population.append(new_solution)

        # Evaluar la nueva población
        new_population_costs = [calculate_total_cost(route, graph) for route in new_population]
        combined_population = population + new_population
        combined_costs = population_costs + new_population_costs

        # Seleccionar las mejores soluciones
        sorted_indices = np.argsort(combined_costs)
        population = [combined_population[i] for i in sorted_indices[:max_population]]
        population_costs = [combined_costs[i] for i in sorted_indices[:max_population]]

        # Actualizar la mejor solución encontrada
        current_best_cost = min(population_costs)
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_routes = population[np.argmin(population_costs)]

    return best_routes, best_cost

def initialize_routes_nearest_neighbor(graph, num_vehicles, vehicle_capacity):
    """
    Inicializa rutas para los vehículos utilizando el algoritmo de Vecino Más Cercano.

    Parámetros:
    - graph: Grafo de la ciudad con arcos que representan las calles y nodos que representan las intersecciones.
    - num_vehicles: Número de vehículos disponibles para la ruta.
    - vehicle_capacity: Capacidad máxima de cada vehículo para recolectar demanda.

    Retorna:
    - routes: Lista de rutas, cada una correspondiente a un vehículo.
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
    """
    Calcula el costo total de todas las rutas generadas.

    Parámetros:
    - routes: Lista de rutas, cada una correspondiente a un vehículo.
    - graph: Grafo de la ciudad que contiene información de las aristas (longitud).

    Retorna:
    - total_cost: Costo total de todas las rutas en términos de longitud acumulada.
    """
    total_cost = 0
    for route in routes:
        for u, v in route:
            if graph.has_edge(u, v):
                edge_data = graph.get_edge_data(u, v)
                total_cost += edge_data[0]['length']
    return total_cost


def mutate_solution(solution, graph, vehicle_capacity):
    """
    Genera una mutación en una solución existente cambiando las rutas.

    Parámetros:
    - solution: Solución actual (rutas).
    - graph: Grafo de la ciudad.
    - vehicle_capacity: Capacidad máxima de cada vehículo.

    Retorna:
    - new_solution: Nueva solución mutada.
    """
    new_solution = [route[:] for route in solution]
    vehicle_id = random.choice([i for i in range(len(new_solution)) if new_solution[i]])
    if len(new_solution[vehicle_id]) > 1:
        idx1, idx2 = random.sample(range(len(new_solution[vehicle_id])), 2)
        new_solution[vehicle_id][idx1], new_solution[vehicle_id][idx2] = new_solution[vehicle_id][idx2], \
        new_solution[vehicle_id][idx1]
    return new_solution

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

    plt.title("Vehicle Routes")
    plt.savefig('all_vehicle_routes.png')
    plt.show()
    plt.close()


# Ejecución principal usando IWO
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

    # Ejecutar IWO para encontrar las mejores rutas
    best_routes, best_cost = iwo_algorithm(graph, num_vehicles, vehicle_capacity)

    print(f"Best total cost: {best_cost}")
    for i, route in enumerate(best_routes):
        print(f"Vehicle {i + 1} Route: {route}")

    plot_all_routes(graph, best_routes)


if __name__ == "__main__":
    main_iwo()
