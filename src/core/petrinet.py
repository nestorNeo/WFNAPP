from dataclasses import dataclass
from typing import Set, Tuple, Union, List, Dict

@dataclass(frozen=True)
class Place:
    id: str

@dataclass(frozen=True)
class Transition:
    id: str  # ID interno
    label: str = None  #  id como etiqueta

@dataclass
class WFN:
    places: Set[Place]
    transitions: Set[Transition]
    arcs: Set[Tuple[Union[Place, Transition], Union[Place, Transition]]]

@dataclass
class Iteration:
    start_event: str
    end_event: str
    sequence: List[str]

def get_traces(input_text):
    return [line.strip().split() for line in input_text.split('\n') if line.strip()]

def detect_concurrency(traces):
    concurrent_relations = set()
    events_positions = {}

    # Paso 1: Registrar las posiciones de cada evento en todas las trazas
    for trace_idx, trace in enumerate(traces):
        for pos, event in enumerate(trace):
            if event not in events_positions:
                events_positions[event] = []
            events_positions[event].append((trace_idx, pos))

    # Paso 2: Analizar las posiciones relativas de los eventos
    for event1 in events_positions:
        for event2 in events_positions:
            if event1 != event2:
                relative_positions = []
                for trace_idx in range(len(traces)):
                    pos1 = next((pos for t, pos in events_positions[event1] if t == trace_idx), None)
                    pos2 = next((pos for t, pos in events_positions[event2] if t == trace_idx), None)
                    if pos1 is not None and pos2 is not None:
                        if abs(pos1 - pos2) == 1:  # Considera eventos adyacentes
                            relative_positions.append(pos1 - pos2)

                # Si los eventos aparecen en orden diferente en distintas trazas y son adyacentes, son concurrentes
                if len(set(relative_positions)) > 1:
                    concurrent_relations.add(tuple(sorted([event1, event2])))

    return concurrent_relations

def group_concurrency(traces, concurrent_relations):
    modified_traces = []
    for trace in traces:
        new_trace = []
        skip_next = False
        for i in range(len(trace)):
            if skip_next:
                skip_next = False
                continue
            if i < len(trace) - 1 and tuple(sorted([trace[i], trace[i+1]])) in concurrent_relations:
                new_trace.append(f"[{min(trace[i], trace[i+1])}||{max(trace[i], trace[i+1])}]")
                skip_next = True
            else:
                new_trace.append(trace[i])
        modified_traces.append(new_trace)
    return modified_traces

def remove_iterations(trace):
    """
    Identifica y extrae todos los tipos de iteraciones:
    1. Repeticiones simples (f f f f)
    2. Pares repetidos (d e d e)
    3. Ciclos tipo (a h k a)
    4. Ciclos complejos
    """
    new_trace = []
    iterations = []
    i = 0
    
    while i < len(trace):
        # Caso 1: Repeticiones simples consecutivas
        if i < len(trace) - 1 and trace[i] == trace[i+1]:
            event = trace[i]
            count = 1
            while i + count < len(trace) and trace[i + count] == event:
                count += 1
            iteration = Iteration(
                start_event=event,
                end_event=event,
                sequence=['epsilon']
            )
            iterations.append(iteration)
            new_trace.append(event)
            i += count
            continue
            
        # Caso 2: Par repetido (d e d e)
        if i < len(trace) - 3:
            if trace[i:i+2] == trace[i+2:i+4]:
                iteration = Iteration(
                    start_event=trace[i],
                    end_event=trace[i+1],
                    sequence=[trace[i], trace[i+1]]
                )
                iterations.append(iteration)
                new_trace.extend(trace[i:i+2])
                i += 4
                continue
        
        # Caso 3: Ciclo tipo (a h k a)
        if i < len(trace) - 2:
            j = i + 2
            while j < len(trace):
                if trace[j] == trace[i]:
                    iteration = Iteration(
                        start_event=trace[i],
                        end_event=trace[i],
                        sequence=trace[i+1:j]
                    )
                    iterations.append(iteration)
                    new_trace.append(trace[i])
                    i = j + 1
                    break
                j += 1
            if j < len(trace):  # Si encontramos un ciclo
                continue
        
        # No hay iteración en esta posición
        new_trace.append(trace[i])
        i += 1

    return new_trace, iterations

def get_precedence_relation(traces):
    precedence = set()
    for trace in traces:
        precedence.add(('start', trace[0]))
        for i in range(len(trace) - 1):
            if trace[i] != trace[i+1]:
                precedence.add((trace[i], trace[i+1]))
        precedence.add((trace[-1], 'end'))
    return precedence

def compute_transitive_closure(traces, precedence_without_iterations):
    closure = set(precedence_without_iterations)
    
    # Agrupar trazas por alfabeto
    trace_groups = {}
    for trace in traces:
        alphabet = frozenset(trace)
        if alphabet not in trace_groups:
            trace_groups[alphabet] = []
        trace_groups[alphabet].append(trace)
    
    # Calcular clausura transitiva para cada grupo
    for group in trace_groups.values():
        group_closure = set()
        for trace in group:
            for i in range(len(trace)):
                for j in range(i + 1, len(trace)):
                    if trace[i] != trace[j]:  # Evitar autolazos
                        group_closure.add((trace[i], trace[j]))
        closure.update(group_closure)
    
    return closure

def should_skip_bridge(a, b, traces):
    """
    Determina si un bridge debe ser omitido:
    1. No bridges si las trazas empiezan igual y el bridge viene del inicio
    2. No bridges desde una agrupación a una subtraza compartida
    3. No bridges si los caminos van a agrupaciones diferentes
    4. No bridges entre subtrazas compartidas
    """
    # Función auxiliar para verificar si un evento es parte de una subtraza compartida
    def is_shared_event(event, pos, traces):
        return all(pos < len(t) and t[pos] == event for t in traces)

    # Verificar si ambos eventos son parte de subtrazas compartidas
    for trace in traces:
        try:
            pos_a = trace.index(a)
            pos_b = trace.index(b)
            if is_shared_event(a, pos_a, traces) and is_shared_event(b, pos_b, traces):
                return True
        except ValueError:
            continue
        
    # Verificar si todas las trazas empiezan igual
    first_events = set(trace[0] for trace in traces)
    if len(first_events) == 1:
        # Si es un bridge desde el inicio, omitirlo
        first_event = next(iter(first_events))
        if a == first_event:
            return True
            
    def is_shared_event(event, pos, traces):
        return all(pos < len(t) and t[pos] == event for t in traces)

    def leads_to_different_groups(pos_a, traces):
        next_groups = set()
        for t in traces:
            if pos_a + 1 < len(t):
                # Obtener el siguiente evento
                next_event = t[pos_a + 1]
                # Buscar la siguiente agrupación después de este evento
                for i in range(pos_a + 1, len(t)):
                    if '||' in str(t[i]):
                        next_groups.add(t[i])
                        break
        return len(next_groups) > 1

    # Caso 1: Si 'a' es una agrupación y 'b' es parte de una subtraza compartida
    if '||' in str(a):
        for trace in traces:
            try:
                pos_a = trace.index(a)
                pos_b = trace.index(b)
                if is_shared_event(b, pos_b, traces):
                    return True
            except ValueError:
                continue

    # Caso 2: Si viene después de una agrupación y va a diferentes agrupaciones
    if not '||' in str(a):  # Si no es una agrupación
        for i, trace in enumerate(traces):
            try:
                pos_a = trace.index(a)
                # Verificar si viene después de una agrupación
                if pos_a > 0 and '||' in str(trace[pos_a - 1]):
                    # Verificar si lleva a diferentes agrupaciones
                    if leads_to_different_groups(pos_a, traces):
                        return True
            except ValueError:
                continue
    
    return False

def simplify_extended_relation(precedence_relation, transitive_closure, traces):
    simplified_relation = set(precedence_relation)
    bridges = set()

    def find_all_shared_subtraces(traces):
        """Encuentra todas las subtrazas compartidas en el conjunto de trazas"""
        shared_subtraces = []
        min_len = min(len(trace) for trace in traces)
        
        start_idx = 0
        while start_idx < min_len:
            # Buscar inicio de subtraza compartida
            events = set(trace[start_idx] for trace in traces)
            if len(events) == 1:
                # Encontramos inicio potencial
                end_idx = start_idx
                while end_idx + 1 < min_len:
                    next_events = set(trace[end_idx + 1] for trace in traces)
                    if len(next_events) > 1:
                        break
                    end_idx += 1
                
                if end_idx > start_idx:
                    shared_subtraces.append((start_idx, end_idx))
                start_idx = end_idx + 1
            else:
                start_idx += 1
                
        return shared_subtraces
    
    def has_divergent_concurrent(traces):
        """Verifica si una agrupación lleva a diferentes caminos sin convergencia"""
        for i, trace in enumerate(traces):
            for j, event in enumerate(trace):
                if '||' in str(event):
                    # Recolectar eventos siguientes
                    next_events = set()
                    for t in traces:
                        if j + 1 < len(t):  # Verificar que el índice es válido
                            next_events.add(t[j + 1])
                    
                    # Si no hay eventos siguientes o solo hay uno, continuar
                    if len(next_events) <= 1:
                        continue
                        
                    # Verificar si estos eventos van a diferentes caminos
                    has_convergence = False
                    common_event = None
                    
                    # Buscar si hay un evento común después
                    for k in range(j + 2, min(len(t) for t in traces)):
                        events_at_k = set(t[k] for t in traces if k < len(t))
                        if len(events_at_k) == 1:
                            has_convergence = True
                            common_event = next(iter(events_at_k))
                            break
                    
                    # Si no hay convergencia, es un patrón que no queremos
                    if not has_convergence:
                        return True
                        
        return False

    def find_all_bifurcations_convergences(traces):
        """Encuentra todos los puntos de bifurcación y convergencia"""
        all_patterns = []
        current_len = 0
        
        while current_len < min(len(t) for t in traces):
            bifurcations = []
            convergences = []
            
            for i in range(current_len + 1, min(len(t) for t in traces)):
                events = set(trace[i] for trace in traces if i < len(trace))
                prev_events = set(trace[i-1] for trace in traces if i-1 < len(trace))
                
                if len(prev_events) == 1 and len(events) > 1:
                    bifurcations.append(i-1)
                elif len(prev_events) > 1 and len(events) == 1:
                    convergences.append(i)
            
            if bifurcations and convergences:
                all_patterns.append((bifurcations, convergences))
                current_len = max(convergences) + 1
            else:
                break
                
        return all_patterns

    def get_concurrent_patterns(traces):
        """Identifica todos los patrones de concurrencia"""
        patterns = []
        for trace in traces:
            current_pattern = []
            for i in range(len(trace)):
                if '||' in trace[i]:
                    if i > 0 and i + 1 < len(trace):
                        current_pattern.append((i-1, i, i+1))
                if current_pattern:
                    patterns.append(current_pattern)
                    current_pattern = []
        return patterns

    # 1. Encontrar todas las subtrazas compartidas
    shared_subtraces = find_all_shared_subtraces(traces)
    for start, end in shared_subtraces:
        for trace in traces:
            if start > 0 and end < len(trace) - 1:
                bridges.add((trace[start-1], trace[end+1]))

    # 2. Procesar todos los patrones de bifurcación/convergencia
    patterns = find_all_bifurcations_convergences(traces)

    # Solo crear bridges en casos válidos
    if not has_divergent_concurrent(traces):
        for bifurcations, convergences in patterns:
            for b in bifurcations:
                for c in convergences:
                    if b < c:
                        for trace in traces:
                            if b < len(trace) and c < len(trace):
                                # Verifica que no hay agrupaciones entre bifurcación y convergencia
                                has_concurrent = False
                                for i in range(b + 1, c):
                                    if i < len(trace) and '||' in str(trace[i]):
                                        has_concurrent = True
                                        break
                                if not has_concurrent:
                                    bridges.add((trace[b], trace[c]))

    # 3. Procesar todos los patrones de concurrencia
    concurrent_patterns = get_concurrent_patterns(traces)
    for pattern_group in concurrent_patterns:
        for pre_idx, conc_idx, post_idx in pattern_group:
            for trace in traces:
                if pre_idx < len(trace) and post_idx < len(trace):
                    bridges.add((trace[pre_idx], trace[post_idx]))

    # Filtra puentes válidos
    valid_bridges = set()
    for a, b in bridges:
        if (a, b) in transitive_closure and (a, b) not in simplified_relation:
            if should_skip_bridge(a, b, traces):
                continue
                
            is_valid = True
            for start, end in shared_subtraces:
                if any(start <= traces[i].index(a) < traces[i].index(b) <= end 
                      for i in range(len(traces)) if a in traces[i] and b in traces[i]):
                    is_valid = False
                    break
            if is_valid:
                valid_bridges.add((a, b))
    
    # Detectar bridges adicionales de concurrencia
    concurrent_bridges = detect_bridges_around_concurrent(simplified_relation, 
                                                       {e for e in simplified_relation if '||' in str(e[0]) or '||' in str(e[1])})
    
    # Filtra también los bridges de concurrencia
    for a, b in concurrent_bridges:
        if (a, b) in transitive_closure and (a, b) not in simplified_relation:
            if not should_skip_bridge(a, b, traces):
                valid_bridges.add((a, b))
    
    # Mantener solo los bridges válidos y añadirlos a la relación simplificada
    simplified_relation.update(valid_bridges)
    
    return simplified_relation, valid_bridges

def get_branching_events(precedence_relation):
    event_counts = {}
    for a, b in precedence_relation:
        if a != 'start' and b != 'end':
            event_counts[a] = event_counts.get(a, 0) + 1
    return {event for event, count in event_counts.items() if count > 1}

def get_events_after_branching(precedence_relation, branching_events):
    return {b for a, b in precedence_relation if a in branching_events}


def analyze_precedence(traces):
    # 1. Primero remover iteraciones
    processed_traces = []
    all_iterations = []
    for trace in traces:
        processed_trace, iterations = remove_iterations(trace)
        processed_traces.append(processed_trace)
        all_iterations.extend(iterations)
        if iterations:
            print(f"Iteraciones removidas de {trace}: {iterations}")
    
    # 2. Detectar concurrencia en las trazas sin iteraciones
    concurrent_relations = detect_concurrency(processed_traces)
    
    # 3. Agrupar la concurrencia
    modified_traces = group_concurrency(processed_traces, concurrent_relations)
    
    # 4. Obtener relaciones de precedencia sin iteraciones
    precedence_without_iterations = get_precedence_relation(modified_traces)
    
    # 5. Para referencia, obtener relaciones con iteraciones usando trazas originales
    original_modified = group_concurrency(traces, concurrent_relations)
    precedence_with_iterations = get_precedence_relation(original_modified)
    
    return (precedence_with_iterations, 
            precedence_without_iterations,
            concurrent_relations, 
            processed_traces, 
            all_iterations)

def analyze_with_transitive_closure(traces):
    # 1. Remover iteraciones y obtener relaciones incluyendo concurrencias
    precedence_with_iterations, precedence_without_iterations, concurrent_relations, processed_traces, iterations = analyze_precedence(traces)
    
    # 2. Aplicar las concurrencias a las trazas procesadas
    processed_traces = group_concurrency(processed_traces, concurrent_relations)
    
    # 3. Calcular cerradura transitiva considerando concurrencias
    transitive_closure = compute_transitive_closure(processed_traces, precedence_without_iterations)
    
    return transitive_closure, processed_traces, precedence_without_iterations

def analyze_with_simplification(traces):
    # 1. Remover iteraciones y obtener relaciones incluyendo concurrencias  
    precedence_with_iterations, precedence_without_iterations, concurrent_relations, processed_traces, iterations = analyze_precedence(traces)
    
    # 2. Aplicar las concurrencias a las trazas procesadas
    processed_traces = group_concurrency(processed_traces, concurrent_relations)
    
    # 3. Calcular cerradura transitiva con trazas que incluyen concurrencias
    transitive_closure = compute_transitive_closure(processed_traces, precedence_without_iterations)
    
    # 4. Calcular relación simplificada usando trazas con concurrencias
    simplified_relation, bridges = simplify_extended_relation(
        precedence_without_iterations,
        transitive_closure,
        processed_traces
    )
    
    return simplified_relation, processed_traces, precedence_without_iterations

def detect_bridges_around_concurrent(simplified_relation, concurrent_events):
    """
    Versión mejorada que detecta bridges alrededor de eventos concurrentes
    """
    bridges = set()
    
    # Mapear eventos anteriores y posteriores a concurrencias
    pre_concurrent = {}  # {concurrent_event: set(previous_events)}
    post_concurrent = {} # {concurrent_event: set(next_events)}
    
    for a, b in simplified_relation:
        if '||' in str(b):
            if b not in pre_concurrent:
                pre_concurrent[b] = set()
            pre_concurrent[b].add(a)
            
        if '||' in str(a):
            if a not in post_concurrent:
                post_concurrent[a] = set()
            post_concurrent[a].add(b)
    
    # Analizar cada evento concurrente
    for conc_event in set(pre_concurrent.keys()) | set(post_concurrent.keys()):
        # Obtener eventos anteriores y posteriores
        prev_events = pre_concurrent.get(conc_event, set())
        next_events = post_concurrent.get(conc_event, set())
        
        # Crear bridges entre eventos anteriores y posteriores
        for prev in prev_events:
            for next in next_events:
                if prev != 'start' and next != 'end':
                    bridges.add((prev, next))
                    
    return bridges

def get_bridge_destinations(a, concurrent_parts, bridge_relations):
    """
    Obtiene los destinos de los bridges para un evento dado y sus partes concurrentes
    """
    destinations = set()
    
    for _, dest in bridge_relations:
        if dest not in concurrent_parts:
            destinations.add(dest)
            
    return destinations

def ungroup_concurrent_event(event: str) -> List[str]:
    """
    Desagrupa un evento concurrente en sus eventos individuales.
    Maneja casos como:
    - [a||b] -> ['a', 'b']
    - [a||b||c] -> ['a', 'b', 'c']
    - evento simple -> [evento]
    """
    if '||' in event:
        # Remueve corchetes y dividir por ||
        parts = event.strip('[]').split('||')
        # Limpia espacios y validar partes
        return [part.strip() for part in parts if part.strip()]
    return [event]

def handle_iterations_robustly(wfn: WFN, iterations: List[Iteration]) -> None:
    """
    Maneja tres tipos específicos de iteraciones:
    1. Repeticiones simples (f f f) -> usar epsilon
    2. Pares repetidos (d e d e) -> usar epsilon
    3. Ciclos con concurrencias ([a||b] c [a||b]) -> manejar desagrupando
    """
    def ungroup_event(event: str) -> List[str]:
        """Desagrupa un evento concurrente o retorna el evento simple"""
        if '||' in event:
            return event.strip('[]').split('||')
        return [event]

    for iteration in iterations:
        # Verificar si es una iteración con epsilon
        has_epsilon = 'epsilon' in iteration.sequence
        
        # 1. CASO: Repetición simple (f f f)
        if len(iteration.sequence) == 1 and has_epsilon:
            # Desagrupar por si hay concurrencia
            start_events = ungroup_event(iteration.start_event)
            
            for event in start_events:
                t = next((t for t in wfn.transitions if t.id == event), None)
                if not t:
                    t = Transition(event)
                    wfn.transitions.add(t)

                # Encontrar lugares de entrada
                pre_places = {p for p, t_out in wfn.arcs if t_out == t}
                
                # Crear ciclo con epsilon
                cycle_place = Place(f'p_cycle_{event}')
                wfn.places.add(cycle_place)
                epsilon = Transition('ε')
                wfn.transitions.add(epsilon)
                
                # Conectar estructura
                wfn.arcs.add((t, cycle_place))
                wfn.arcs.add((cycle_place, epsilon))
                for pre_place in pre_places:
                    wfn.arcs.add((epsilon, pre_place))

        # 2. CASO: Par repetido (d e d e)
        elif len(iteration.sequence) == 2 and iteration.sequence[0] == iteration.start_event:
            start_events = ungroup_event(iteration.start_event)
            end_events = ungroup_event(iteration.end_event)
            
            for s_event, e_event in zip(start_events, end_events):
                t1 = next((t for t in wfn.transitions if t.id == s_event), None)
                t2 = next((t for t in wfn.transitions if t.id == e_event), None)
                
                if not t1:
                    t1 = Transition(s_event)
                    wfn.transitions.add(t1)
                if not t2:
                    t2 = Transition(e_event)
                    wfn.transitions.add(t2)

                # Encontrar lugares de entrada
                pre_places = {p for p, t_out in wfn.arcs if t_out == t1}
                
                # Crear estructura del ciclo
                cycle_place = Place(f'p_cycle_{s_event}_{e_event}')
                inter_place = Place(f'p_{s_event}_{e_event}')
                wfn.places.add(cycle_place)
                wfn.places.add(inter_place)
                epsilon = Transition('ε')
                wfn.transitions.add(epsilon)
                
                # Conectar estructura
                wfn.arcs.add((t1, inter_place))
                wfn.arcs.add((inter_place, t2))
                wfn.arcs.add((t2, cycle_place))
                wfn.arcs.add((cycle_place, epsilon))
                for pre_place in pre_places:
                    wfn.arcs.add((epsilon, pre_place))

        # 3. CASO: Ciclos con concurrencias
        else:
            # Desagrupar todos los eventos
            start_events = ungroup_event(iteration.start_event)
            sequence_events = []
            for event in iteration.sequence:
                sequence_events.extend(ungroup_event(event))

            # Manejar cada evento desagrupado
            prev_trans = {t for t in wfn.transitions if t.id in start_events}
            if not prev_trans:
                for event in start_events:
                    t = Transition(event)
                    wfn.transitions.add(t)
                    prev_trans.add(t)

            # CAMBIO AQUÍ: Manejar la estructura secuencial con un solo lugar para bifurcaciones
            for event in sequence_events:
                curr_t = next((t for t in wfn.transitions if t.id == event), None)
                if not curr_t:
                    curr_t = Transition(event)
                    wfn.transitions.add(curr_t)
                
                # Para cada transición previa, crear o reutilizar un solo lugar
                for prev_t in prev_trans:
                    # Buscar si ya existe un lugar de salida para prev_t
                    existing_place = None
                    for arc in wfn.arcs:
                        if isinstance(arc[0], Transition) and arc[0] == prev_t and isinstance(arc[1], Place):
                            existing_place = arc[1]
                            break
                    
                    if not existing_place:
                        # Si no existe, crear un nuevo lugar
                        existing_place = Place(f'p_cycle_{prev_t.id}')
                        wfn.places.add(existing_place)
                        wfn.arcs.add((prev_t, existing_place))
                    
                    # Usar el lugar existente para conectar con la nueva transición
                    wfn.arcs.add((existing_place, curr_t))
                
                prev_trans = {curr_t}

            # Conectar con los lugares originales usando el mismo principio
            for prev_t in prev_trans:
                for start_t in start_events:
                    t = next((t for t in wfn.transitions if t.id == start_t), None)
                    if t:
                        pre_places = {p for p, t_out in wfn.arcs if t_out == t}
                        for pre_place in pre_places:
                            wfn.arcs.add((prev_t, pre_place))

    return wfn


def handle_concurrent_relation(wfn: WFN, a: str, b: str) -> None:
    """
    Maneja relaciones con eventos concurrentes INTERMEDIOS (no en inicio/fin).
    """
    print(f"Debug handle_concurrent_relation: a = {a}, b = {b}")

    def get_or_create_transition(event_id: str) -> Transition:
        """Obtiene o crea una transición única para el evento dado"""
        t = next((t for t in wfn.transitions if t.id == event_id), None)
        if t is None:
            t = Transition(event_id)
            wfn.transitions.add(t)
            print(f"Transition created: {t.id}")
        return t

    def get_or_create_place(event: str, place_type: str) -> Place:
        """Obtiene o crea un lugar único basado en el tipo y el evento"""
        place_id = f'p_{place_type}_{event}'
        p = next((p for p in wfn.places if p.id == place_id), None)
        if p is None:
            p = Place(place_id)
            wfn.places.add(p)
            print(f"Place created: {p.id}")
        return p

    # Esta función solo maneja concurrencia INTERMEDIA
    if '||' in a and b != 'end' and a != 'start':
        # a es concurrente, b es simple
        concurrent_events = a.strip('[]').split('||')
        
        # Crear lugares de join para la concurrencia
        join_places = {}
        for e in concurrent_events:
            t = get_or_create_transition(e)
            join_place = get_or_create_place(e, "join")
            join_places[e] = join_place
            wfn.arcs.add((t, join_place))
        
        # Conectar con el siguiente evento
        other_transition = get_or_create_transition(b)
        for join_place in join_places.values():
            wfn.arcs.add((join_place, other_transition))
            
    elif '||' in b and a != 'start' and b != 'end':
        # a es simple, b es concurrente
        concurrent_events = b.strip('[]').split('||')
        
        # Crear lugares de split para la concurrencia
        split_places = {}
        for e in concurrent_events:
            split_place = get_or_create_place(e, "split")
            split_places[e] = split_place
            t = get_or_create_transition(e)
            wfn.arcs.add((split_place, t))
        
        # Conectar desde el evento anterior
        other_transition = get_or_create_transition(a)
        for split_place in split_places.values():
            wfn.arcs.add((other_transition, split_place))

    print("Exiting handle_concurrent_relation")

def find_shared_sequences(precedence_relation):
    """
    Encuentra secuencias de eventos que son compartidas entre diferentes trazas.
    Por ejemplo, si tenemos trazas:
    x a b c d y
    w e b c f z
    Detectará que 'b c' es una secuencia compartida
    """
    shared_sequences = set()
    sequence_entries = {} # Para almacenar las entradas a cada secuencia
    sequence_exits = {}   # Para almacenar las salidas de cada secuencia
    
    # Buscar secuencias de al menos 2 eventos que aparecen en el mismo orden
    current_sequence = []
    for a, b in precedence_relation:
        if a != 'start' and b != 'end':
            # Buscar secuencias que empiezan con este par
            sequence = []
            sequence.append(a)
            sequence.append(b)
            
            # Buscar continuaciones de la secuencia
            next_event = b
            while True:
                next_relations = [(x, y) for x, y in precedence_relation if x == next_event and y != 'end']
                if len(next_relations) != 1:
                    break
                next_event = next_relations[0][1]
                sequence.append(next_event)
                
            if len(sequence) >= 2:
                sequence_tuple = tuple(sequence)
                shared_sequences.add(sequence_tuple)
                
                # Registrar entradas y salidas
                if sequence[0] not in sequence_entries:
                    sequence_entries[sequence[0]] = set()
                sequence_entries[sequence[0]].add(sequence_tuple)
                
                if sequence[-1] not in sequence_exits:
                    sequence_exits[sequence[-1]] = set()
                sequence_exits[sequence[-1]].add(sequence_tuple)

    return shared_sequences, sequence_entries, sequence_exits

def is_shared_sequence(a, b, shared_info):
    """
    Verifica si una relación (a,b) es parte de una secuencia compartida
    """
    shared_sequences, sequence_entries, sequence_exits = shared_info
    
    # Buscar si esta relación es parte de alguna secuencia compartida
    for sequence in shared_sequences:
        # Buscar si a y b aparecen consecutivamente en la secuencia
        for i in range(len(sequence)-1):
            if sequence[i] == a and sequence[i+1] == b:
                return True
                
    return False

def construct_wfn(simplified_relation, precedence_relation, concurrent_groups, iterations, bridges, include_iterations=False):
    print("Constructing WFN...")
    wfn = WFN(places=set(), transitions=set(), arcs=set())

    start_place = Place('start')
    end_place = Place('end')
    wfn.places.add(start_place)
    wfn.places.add(end_place)
    
    # NO crear lugares intermedios después de start o antes de end
    
    # Procesar todas las relaciones
    for a, b in simplified_relation:
        print(f"Processing relation: {a} -> {b}")
        
        # Caso especial: inicio
        if a == 'start':
            if '||' in b:
                # Evento concurrente al inicio - necesita epsilon y estructura especial
                concurrent_events = b.strip('[]').split('||')
                
                # Crear epsilon inicial
                init_epsilon = get_or_create_transition('ε_init_' + '_'.join(sorted(concurrent_events)), wfn)
                wfn.arcs.add((start_place, init_epsilon))
                
                # Crear lugares de split para cada evento concurrente
                for e in concurrent_events:
                    split_place = Place(f'p_split_{e}')
                    wfn.places.add(split_place)
                    wfn.arcs.add((init_epsilon, split_place))
                    
                    # Crear la transición del evento
                    t = get_or_create_transition(e, wfn)
                    wfn.arcs.add((split_place, t))
            else:
                # Evento simple al inicio - conectar DIRECTAMENTE desde start
                t = get_or_create_transition(b, wfn)
                wfn.arcs.add((start_place, t))
        
        # Caso especial: fin
        elif b == 'end':
            if '||' in a:
                # Evento concurrente al final - necesita epsilon y estructura especial
                concurrent_events = a.strip('[]').split('||')
                
                # Crear epsilon final
                final_epsilon = get_or_create_transition('ε_final_' + '_'.join(sorted(concurrent_events)), wfn)
                wfn.arcs.add((final_epsilon, end_place))
                
                # Crear lugares de join para cada evento concurrente
                for e in concurrent_events:
                    join_place = Place(f'p_join_{e}')
                    wfn.places.add(join_place)
                    wfn.arcs.add((join_place, final_epsilon))
                    
                    # La transición del evento ya debe existir
                    t = get_or_create_transition(e, wfn)
                    wfn.arcs.add((t, join_place))
            else:
                # Evento simple al final - conectar DIRECTAMENTE a end
                t = get_or_create_transition(a, wfn)
                wfn.arcs.add((t, end_place))
        
        # Casos con concurrencia intermedia
        elif '||' in a or '||' in b:
            handle_concurrent_relation(wfn, a, b)
        
        # Caso normal (relación entre dos eventos no especiales)
        else:
            t1 = get_or_create_transition(a, wfn)
            t2 = get_or_create_transition(b, wfn)
            
            # IMPORTANTE: Verificar si esta relación es un bridge
            is_bridge = (a, b) in bridges
            
            # Si es un bridge, crear su propio lugar único sin reutilizar
            if is_bridge:
                p = Place(f'p_{a}_{b}_bridge')
                wfn.places.add(p)
                wfn.arcs.add((t1, p))
                wfn.arcs.add((p, t2))
                print(f"  Created bridge place {p.id} for bridge {a} -> {b}")
            else:
                # Solo aplicar la lógica de unificación si NO es un bridge
                # CORRECCIÓN PARA DIVERGENCIAS: Verificar si t1 ya tiene un lugar de salida
                existing_output_place = None
                for arc in wfn.arcs:
                    # Buscar un lugar que salga de t1
                    if arc[0] == t1 and isinstance(arc[1], Place):
                        # Verificar que este lugar no sea end, no sea de concurrencia, y no sea un bridge
                        if (arc[1].id != 'end' and 
                            not ('split' in arc[1].id or 'join' in arc[1].id) and
                            not ('bridge' in arc[1].id)):
                            existing_output_place = arc[1]
                            break
                
                # CORRECCIÓN PARA CONVERGENCIAS: Verificar si ya existe un lugar que entre a t2
                existing_input_place = None
                for arc in wfn.arcs:
                    # Buscar un lugar que ya conecte con t2
                    if isinstance(arc[0], Place) and arc[1] == t2:
                        # Verificar que este lugar no sea start/end, no sea de concurrencia, y no sea un bridge
                        if (arc[0].id not in ['start', 'end'] and 
                            not ('split' in arc[0].id or 'join' in arc[0].id) and
                            not ('bridge' in arc[0].id)):
                            # Verificar que no haya ya una conexión desde t1 a este lugar
                            already_connected = False
                            for existing_arc in wfn.arcs:
                                if existing_arc[0] == t1 and existing_arc[1] == arc[0]:
                                    already_connected = True
                                    break
                            
                            if not already_connected:
                                existing_input_place = arc[0]
                                break
                
                # Decidir qué hacer basándose en lo que encontramos
                if existing_output_place and existing_input_place and existing_output_place == existing_input_place:
                    # Ya existe la conexión, no hacer nada
                    print(f"  Connection already exists between {a} and {b}")
                elif existing_output_place:
                    # Reutilizar el lugar de salida existente (punto de divergencia)
                    wfn.arcs.add((existing_output_place, t2))
                    print(f"  Reusing existing output place {existing_output_place.id} for divergence from {a}")
                elif existing_input_place:
                    # Reutilizar el lugar de entrada existente (punto de convergencia)
                    wfn.arcs.add((t1, existing_input_place))
                    print(f"  Reusing existing input place {existing_input_place.id} for convergence to {b}")
                else:
                    # Crear un nuevo lugar si no existe ninguno
                    p = Place(f'p_{a}_{b}')
                    wfn.places.add(p)
                    wfn.arcs.add((t1, p))
                    wfn.arcs.add((p, t2))
                    print(f"  Created new place {p.id}")

    # Agregar iteraciones si está habilitado
    if include_iterations:
        handle_iterations_robustly(wfn, iterations)

    # Validación final de la estructura de WFN
    print("Final structure of WFN:")
    print(f"Places: {[place.id for place in wfn.places]}")
    print(f"Transitions: {[transition.id for transition in wfn.transitions]}")
    print(f"Arcs: {[(arc[0].id, arc[1].id) for arc in wfn.arcs]}")
    
    return wfn

def get_or_create_transition(event_id: str, wfn: WFN) -> Transition:
    """Helper function para obtener o crear una transición"""
    t = next((t for t in wfn.transitions if t.id == event_id), None)
    if not t:
        t = Transition(event_id)
        wfn.transitions.add(t)
        print(f"Transition created: {t}")
    else:
        print(f"Transition already exists: {t}")
    return t

def get_or_create_place(event_id: str, wfn: WFN) -> Place:
    """Helper function para obtener o crear un lugar"""
    p = next((p for p in wfn.places if p.id == event_id), None)
    if not p:
        p = Place(event_id)
        wfn.places.add(p)
    return p

def analyze_and_construct_wfn(traces, include_iterations=False):
    """
    Función principal que mantiene la misma interfaz pero usa la nueva construcción.
    """
    precedence_with_iterations, precedence_without_iterations, concurrent_relations, processed_traces, iterations = analyze_precedence(traces)
    transitive_closure, _, _ = analyze_with_transitive_closure(traces)
    simplified_relation, bridges = simplify_extended_relation(precedence_without_iterations, transitive_closure, processed_traces)

    if not include_iterations:
        simplified_relation = {(a, b) for a, b in simplified_relation if a != b}
        precedence_without_iterations = {(a, b) for a, b in precedence_without_iterations if a != b}

    wfn = construct_wfn(
        simplified_relation,
        precedence_with_iterations if include_iterations else precedence_without_iterations,
        concurrent_relations,
        iterations,
        bridges,
        include_iterations
    )

    return wfn, simplified_relation, precedence_without_iterations, bridges