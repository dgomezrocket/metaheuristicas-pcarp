# Metaheurísticas para el Capacitated Arc Routing Problem (CARP)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-orange.svg)](https://networkx.org/)
[![OSMnx](https://img.shields.io/badge/OSMnx-1.3+-green.svg)](https://osmnx.readthedocs.io/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-red.svg)](https://matplotlib.org/)

## Descripción del Proyecto

Este proyecto implementa y compara tres algoritmos metaheurísticos para resolver el Problema de Rutas por Arcos con Capacidad (PCARP - Periodic Capacitated Arc Routing Problem). Este trabajo fue desarrollado como parte de un curso de Computación Evolutiva a nivel de Maestría.

El PCARP es un problema de optimización combinatoria donde se deben encontrar rutas óptimas para una flota de vehículos con capacidad limitada, que deben atender demandas ubicadas en las aristas de un grafo (por ejemplo, calles de una ciudad).

## Algoritmos Implementados

Este proyecto implementa tres metaheurísticas competitivas del estado del arte:

1. **Variable Neighborhood Search (VNS)**: Explora sistemáticamente distintas vecindades para escapar de óptimos locales.
2. **Invasive Weed Optimization (IWO)**: Algoritmo bio-inspirado en la colonización de malezas.
3. **Simulated Annealing (SA)**: Técnica probabilística que simula el proceso de recocido en metalurgia.

## Características del Proyecto

- 🌍 Utiliza datos reales de OpenStreetMap a través de OSMnx
- 🚗 Implementa soluciones para el problema de enrutamiento de vehículos con capacidad
- 📊 Compara el rendimiento de múltiples algoritmos metaheurísticos
- 📈 Incluye visualizaciones animadas de las soluciones generadas
- 🧪 Aplica metodología científica rigurosa para la evaluación comparativa

## Estructura del Proyecto

```
├── algoritmos/
│   ├── vns.py                # Implementación del algoritmo Variable Neighborhood Search
│   ├── iwo.py                # Implementación del algoritmo Invasive Weed Optimization
│   └── simulatedannealing.py # Implementación del algoritmo Simulated Annealing
│
├── visualizacion/
│   ├── vns_DibujarRutas.py   # Visualización animada de rutas obtenidas con VNS
│   ├── iwo_DibujoRutas.py    # Visualización animada de rutas obtenidas con IWO
│   └── SA_DibujoRutas.py     # Visualización animada de rutas de Simulated Annealing
│
├── experimentos/
│   ├── comparativa.py        # Script principal para comparar el rendimiento de los algoritmos
│   └── benchmarks/           # Instancias del problema empleadas para evaluar los algoritmos
│
├── docs/
│   ├── CARP_Derlis_Gomez_v3.pdf        # Presentación técnica del trabajo
│   ├── IMRAD_PCARP_DERLIS_GOMEZ_v3.pdf # Reporte técnico en formato IMRAD
│   └── imagenes/                       # Imágenes y diagramas utilizados en la documentación
│
├── .gitignore                # Lista de archivos/directorios que Git debe ignorar
├── requirements.txt          # Dependencias de Python necesarias para ejecutar el proyecto
└── README.md                 # Este archivo de documentación
```

## Resultados Visuales

El proyecto genera visualizaciones de las rutas optimizadas para cada algoritmo, mostrando las trayectorias de los vehículos en un mapa real de la ciudad. Además, incluye animaciones que ilustran el proceso de construcción de las rutas para una mejor comprensión del funcionamiento de los algoritmos.

## Resultados Clave

Las principales conclusiones del estudio comparativo son:

- El algoritmo VNS ofrece el mejor equilibrio entre calidad de solución y tiempo de ejecución
- IWO es más efectivo cuando se requiere exploración amplia del espacio de soluciones
- SA proporciona resultados consistentes con menos varianza entre ejecuciones

## Cómo Ejecutar

1. Clonar el repositorio:
bash git clone [https://github.com/dgomezrocket/metaheuristicas-pcarp.git](https://github.com/dgomezrocket/metaheuristicas-pcarp.git) cd metaheuristicas-pcarp

2. Instalar dependencias:

bash pip install -r requirements.txt

3. Ejecutar el algoritmo deseado:

bash python algoritmos/vns.py  Para ejecutar VNS 

python algoritmos/iwo.py Para ejecutar IWO 

python algoritmos/simulatedannealing.py Para ejecutar SA

4. Para visualizar las rutas con animaciones:

bash python visualizacion/vns_DibujarRutas.py



## Tecnologías Utilizadas

- **Python**: Lenguaje principal de implementación
- **NetworkX**: Biblioteca para la creación y manipulación de grafos
- **OSMnx**: Extracción de datos de OpenStreetMap
- **Matplotlib**: Visualización de datos y animaciones
- **NumPy**: Operaciones numéricas eficientes
- **Scikit-learn**: Algoritmos de clustering para la agrupación inicial

## Metodología

Este proyecto sigue rigurosamente el método científico:

1. **Revisión de Literatura**: Análisis sistemático de los últimos 5 años de investigación en metaheurísticas para CARP.
2. **Selección de Algoritmos**: Identificación de tres enfoques competitivos del estado del arte.
3. **Implementación**: Codificación en Python siguiendo buenas prácticas.
4. **Experimentación**: Evaluación rigurosa de desempeño con métricas estadísticas.
5. **Análisis de Resultados**: Comparación objetiva e interpretación de hallazgos.

## Comparativa de Algoritmos

| Algoritmo | Costo Promedio | Tiempo de Ejecución | Robustez |
|-----------|----------------|---------------------|----------|
| VNS       | Menor          | Medio               | Alta     |
| IWO       | Medio          | Mayor               | Media    |
| SA        | Mayor          | Menor               | Alta     |

## Autor

Derlis Gomez

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

