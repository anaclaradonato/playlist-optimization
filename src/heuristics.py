"""
Módulo de Heurísticas de Otimização e Linha de Base

Responsabilidades:
1. Linha de Base Aleatória (Random Shuffle Baseline): avalia 500 permutações aleatórias.
2. Heurística Construtiva Gulosa (Nearest Neighbor com Multi-Start):
   Gera uma solução inicial conectando a cada passo a música de menor atrito acústico.
3. Busca Local Or-opt (Or, 1976):
   Refina a rota transferindo blocos de 1, 2 ou 3 músicas para melhores posições.
4. Solver Heurístico Integrado:
   Combina Vizinho Mais Próximo + Or-opt para resolução instantânea em frações de segundo.
"""

import time
import random
import numpy as np
from cost_matrix import evaluate_sequence_cost


def random_shuffle_baseline(
    cost_matrix,
    n_shuffles=500,
    seed=0
):
    """
    Gera n_shuffles permutações aleatórias da playlist para estabelecer a linha de base
    estatística (uma referência: a reprodução sem nenhum critério de curadoria).
    
    Retorna a média, o desvio padrão e o melhor custo encontrado no sorteio.
    """
    n = cost_matrix.shape[0]
    rng = random.Random(seed)
    costs = []
    
    base_seq = list(range(n))
    for _ in range(n_shuffles):
        shuffled = base_seq.copy()
        rng.shuffle(shuffled)
        c = evaluate_sequence_cost(shuffled, cost_matrix)
        costs.append(c)
        
    return {
        'mean_cost': float(np.mean(costs)),
        'std_cost': float(np.std(costs)),
        'min_cost': float(np.min(costs)),
        'max_cost': float(np.max(costs))
    }


def nearest_neighbor(
    cost_matrix,
    multi_start=True
):
    """
    Heurística Construtiva do Vizinho Mais Próximo (Nearest Neighbor).
    
    A partir de uma música inicial, seleciona iterativamente a próxima música não visitada
    que minimiza o custo de transição c_ij.
    
    Com multi_start=True, testa cada uma das n faixas como ponto de partida
    e retorna a melhor sequência construída.
    """
    n = cost_matrix.shape[0]
    best_route = []
    best_cost = float('inf')
    
    # Se multi_start for True, testa iniciar em cada uma das n músicas
    starts = range(n) if multi_start else [0]
    
    for start in starts:
        route = [start]
        visited = set(route)
        current = start
        
        while len(route) < n:
            # Encontra a música não visitada com menor custo a partir da atual
            best_next = None
            min_c = float('inf')
            
            for j in range(n):
                if j not in visited:
                    c = cost_matrix[current, j]
                    if c < min_c:
                        min_c = c
                        best_next = j
                        
            route.append(best_next)
            visited.add(best_next)
            current = best_next
            
        cost = evaluate_sequence_cost(route, cost_matrix)
        if cost < best_cost:
            best_cost = cost
            best_route = route
            
    return best_route, best_cost


def or_opt(cost_matrix, initial_route, max_segment_len=3, max_passes=1000):
    """
    Busca Local Or-opt (Or, 1976).
    Refina a rota transferindo blocos de 1 a 3 músicas para melhores posições.
    """
    route = initial_route.copy()
    current_cost = evaluate_sequence_cost(route, cost_matrix)
    n = len(route)
    
    for _ in range(max_passes):
        improved = False
        for seg_len in range(max_segment_len, 0, -1):
            for i in range(n - seg_len + 1):
                seg = route[i:i + seg_len]
                rem = route[:i] + route[i + seg_len:]
                for pos in range(len(rem) + 1):
                    if pos == i:
                        continue
                    cand = rem[:pos] + seg + rem[pos:]
                    cand_cost = evaluate_sequence_cost(cand, cost_matrix)
                    if cand_cost < current_cost - 1e-9:
                        route, current_cost = cand, cand_cost
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
            
    return route, current_cost


def solve_heuristic(cost_matrix):
    """
    Resolve o sequenciamento da playlist utilizando a combinação Heurística Completa:
    1. Constrói uma rota inicial via Vizinho Mais Próximo Multi-Start;
    2. Refina a rota via Busca Local Or-opt (blocos de 1 a 3 faixas).
    
    Retorna o dicionário de resultados com tempo de CPU, rota e custo final.
    """
    t0 = time.time()
    nn_route, nn_cost = nearest_neighbor(cost_matrix, multi_start=True)
    refined_route, final_cost = or_opt(cost_matrix, nn_route, max_segment_len=3)
    
    return {
        'status': 'Heuristic_Local_Optimum',
        'route': refined_route,
        'cost': final_cost,
        'nn_cost': nn_cost,
        'runtime_sec': time.time() - t0,
        'num_tracks': len(refined_route)
    }
