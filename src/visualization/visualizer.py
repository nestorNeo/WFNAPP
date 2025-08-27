from graphviz import Digraph
import os
from src.core.petrinet import WFN, Place, Transition
from collections import deque, defaultdict

def visualize_sequences(traces, filename="sequences"):
    """Visualiza las secuencias de trazas"""
    dot = Digraph(comment='Secuencias')
    dot.attr(rankdir='LR')
    
    dot.node('start', 'start', shape='ellipse')
    dot.node('end', 'end', shape='ellipse')
    
    for i, sequence in enumerate(traces):
        prev_node = 'start'
        for j, event in enumerate(sequence):
            current_node = f"{event}_{i}_{j}"
            dot.node(current_node, event)
            dot.edge(prev_node, current_node)
            prev_node = current_node
        dot.edge(prev_node, 'end')
    
    output_path = os.path.join("output", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dot.render(output_path, format="png", cleanup=True)
    return f"{output_path}.png"

def visualize_precedence(precedence_relations, traces, filename, include_loops=True):
    """Visualiza el grafo de precedencia"""
    dot = Digraph(comment='Relación de precedencia')
    dot.attr(rankdir='LR')

    # Identificar nodos únicos
    nodes = set()
    for a, b in precedence_relations:
        if a != 'start' and a != 'end':
            nodes.add(a)
        if b != 'start' and b != 'end':
            nodes.add(b)

    # Añadir nodos especiales
    dot.node('start', 'start', shape='ellipse')
    dot.node('end', 'end', shape='ellipse')

    # Añadir nodos de eventos
    for node in nodes:
        if '||' in str(node):
            # Evento compuesto
            dot.node(str(node), str(node), shape='box', style='filled', fillcolor='lightgrey')
        else:
            # Evento simple
            dot.node(str(node), str(node), shape='circle')

    # Añadir arcos
    for a, b in precedence_relations:
        if a == b and include_loops:
            # Auto-lazo para iteraciones
            dot.edge(str(a), str(b), label="Iteración", style="dashed", color="red")
        else:
            dot.edge(str(a), str(b))

    output_path = os.path.join("output", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dot.render(output_path, format="png", cleanup=True)
    return f"{output_path}.png"

def visualize_transitive_closure(traces, closure_relations, precedence_without_iterations, filename):
    """Visualiza el cierre transitivo"""
    dot = Digraph(comment='Cierre transitivo')
    dot.attr(rankdir='LR')

    # Identificar todos los eventos únicos
    all_events = set()
    for trace in traces:
        for event in trace:
            all_events.add(event)

    # Crear nodos
    dot.node('start', 'start', shape='ellipse')
    dot.node('end', 'end', shape='ellipse')

    for event in all_events:
        if '||' in str(event):
            dot.node(str(event), str(event), shape='box', style='filled', fillcolor='lightgrey')
        else:
            dot.node(str(event), str(event), shape='circle')

    # Dibujar relaciones de precedencia originales
    for a, b in precedence_without_iterations:
        dot.edge(str(a), str(b), color='black')

    # Dibujar relaciones de cerradura transitiva (solo las nuevas)
    for a, b in closure_relations:
        if (a, b) not in precedence_without_iterations:
            dot.edge(str(a), str(b), color='red', style='dashed')

    output_path = os.path.join("output", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dot.render(output_path, format="png", cleanup=True)
    return f"{output_path}.png"

def visualize_simplified_relation(traces, simplified_relation, bridges, precedence_without_iterations, filename):
    """Visualiza la relación simplificada"""
    dot = Digraph(comment='Relación simplificada')
    dot.attr(rankdir='LR')

    # Crear nodos para todos los eventos únicos
    all_events = set()
    for trace in traces:
        for event in trace:
            all_events.add(event)

    # Añadir nodos especiales
    dot.node('start', 'start', shape='ellipse')
    dot.node('end', 'end', shape='ellipse')

    # Añadir nodos de eventos
    for event in all_events:
        if '||' in str(event):
            dot.node(str(event), str(event), shape='box', style='filled', fillcolor='lightgrey')
        else:
            dot.node(str(event), str(event), shape='circle')

    # Dibujar las relaciones simplificadas
    for a, b in simplified_relation:
        # Los bridges se muestran en rojo punteado
        if (a, b) in bridges:
            dot.edge(str(a), str(b), color='red', style='dashed')
        else:
            dot.edge(str(a), str(b), color='black')

    output_path = os.path.join("output", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dot.render(output_path, format="png", cleanup=True)
    return f"{output_path}.png"

def create_ordered_place_labels(wfn: WFN):
    """
    Crea etiquetas ordenadas para los lugares siguiendo el flujo de la red.
    Los lugares se numeran desde el inicio siguiendo el orden lógico de ejecución.
    """
    place_labels = {}
    
    # Identificar lugares especiales
    start_place = None
    end_place = None
    
    for place in wfn.places:
        if place.id == 'start':
            place_labels[place.id] = 'i'  # Lugar inicial según el paper
            start_place = place
        elif place.id == 'end':
            place_labels[place.id] = 'o'  # Lugar final según el paper  
            end_place = place
    
    # Obtener lugares regulares (excluyendo start y end)
    regular_places = [p for p in wfn.places if p != start_place and p != end_place]
    
    if not regular_places:
        return place_labels
    
    # Crear grafo de adyacencia para ordenamiento topológico
    adjacency = defaultdict(set)
    in_degree = defaultdict(int)
    
    # Inicializar todos los lugares regulares
    for place in regular_places:
        in_degree[place] = 0
    
    # Analizar conexiones para determinar orden
    for source, target in wfn.arcs:
        if isinstance(source, Transition) and isinstance(target, Place):
            # Transición -> Lugar: buscar lugares anteriores
            for prev_source, prev_target in wfn.arcs:
                if (isinstance(prev_source, Place) and isinstance(prev_target, Transition) 
                    and prev_target == source):
                    # Lugar anterior -> Transición -> Lugar actual
                    if (prev_source in regular_places and target in regular_places 
                        and prev_source != target):
                        adjacency[prev_source].add(target)
                        in_degree[target] += 1
    
    # Ordenamiento topológico con BFS
    queue = deque()
    
    # Encontrar lugares sin dependencias (grado de entrada 0)
    for place in regular_places:
        if in_degree[place] == 0:
            queue.append(place)
    
    # Si no hay lugares con grado 0, empezar con los conectados a start
    if not queue:
        for source, target in wfn.arcs:
            if (isinstance(source, Place) and source == start_place and 
                isinstance(target, Transition)):
                # start -> transición, buscar lugares después de esta transición
                for next_source, next_target in wfn.arcs:
                    if (isinstance(next_source, Transition) and next_source == target and
                        isinstance(next_target, Place) and next_target in regular_places):
                        queue.append(next_target)
                        break
    
    # Si aún no hay lugares en la cola, empezar con el primero alfabéticamente
    if not queue and regular_places:
        queue.append(sorted(regular_places, key=lambda p: p.id)[0])
    
    # Asignar números en orden
    place_counter = 1
    visited = set()
    
    while queue:
        current_place = queue.popleft()
        
        if current_place in visited:
            continue
            
        visited.add(current_place)
        place_labels[current_place.id] = f"p{place_counter}"
        place_counter += 1
        
        # Agregar lugares dependientes
        for next_place in adjacency[current_place]:
            if next_place not in visited:
                in_degree[next_place] -= 1
                if in_degree[next_place] == 0:
                    queue.append(next_place)
    
    # Asignar números a lugares restantes
    remaining_places = [p for p in regular_places if p not in visited]
    for place in sorted(remaining_places, key=lambda p: p.id):
        place_labels[place.id] = f"p{place_counter}"
        place_counter += 1
    
    return place_labels

def visualize_wfn(wfn: WFN, bridges: set, filename: str):
    """
    Visualiza la Red de Flujo de Trabajo con lugares numerados ordenadamente.
    """
    dot = Digraph(comment='Workflow Net')
    dot.attr(rankdir='LR')
    
    # Crear etiquetas ordenadas para los lugares
    place_labels = create_ordered_place_labels(wfn)
    
    # Debug: mostrar el mapeo de lugares
    print(f"\n=== Mapeo de Lugares Ordenado ===")
    sorted_labels = sorted([(k, v) for k, v in place_labels.items()], 
                          key=lambda x: (x[1] == 'i', x[1] == 'o', x[1]))
    for place_id, label in sorted_labels:
        print(f"  {place_id} -> {label}")
    
    # Añadir nodos de lugares
    for place in wfn.places:
        visual_label = place_labels[place.id]
        
        if place.id == 'start':
            dot.node(place.id, visual_label, shape='circle', style='filled', 
                    fillcolor='lightblue', width='0.5')
        elif place.id == 'end':
            dot.node(place.id, visual_label, shape='circle', style='filled', 
                    fillcolor='lightcoral', width='0.5')
        else:
            dot.node(place.id, visual_label, shape='circle', style='filled', 
                    fillcolor='white', width='0.4')
    
    # Añadir transiciones
    for transition in wfn.transitions:
        if 'ε' in transition.id or (transition.label and 'ε' in transition.label):
            label = 'ε'
            style = 'filled'
            fillcolor = 'lightgrey'
        else:
            label = transition.label if transition.label else transition.id
            style = 'filled'
            fillcolor = 'white'
        
        dot.node(transition.id, label, shape='box', style=style, 
                fillcolor=fillcolor, height='0.4')
    
    # Añadir arcos
    for source, target in wfn.arcs:
        if isinstance(source, Place) and isinstance(target, Transition):
            dot.edge(source.id, target.id, arrowhead='normal', color='black')
        elif isinstance(source, Transition) and isinstance(target, Place):
            dot.edge(source.id, target.id, arrowhead='normal', color='black')
        else:
            dot.edge(source.id, target.id, style='dashed', color='red')
    
    # Renderizar y guardar
    output_path = os.path.join("output", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dot.render(output_path, format="png", cleanup=True)
    return f"{output_path}.png"