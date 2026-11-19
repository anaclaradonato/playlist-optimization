"""
Módulo Solver Exato Open ATSP no HiGHS (open_atsp_solver.py)
Projeto: Sequenciamento Ótimo de Playlists (Open ATSP) - Trabalho Final de PL

Responsabilidades:
1. Modelar o Caixeiro-Viajante Assimétrico de Caminho Aberto (Open ATSP) em PLIM.
2. Inserção do nó fictício (dummy) com arcos de custo zero para abrir o circuito.
3. Eliminação de sub-rotas via restrições de Miller-Tucker-Zemlin (MTZ).
4. Resolução exata utilizando o solver HiGHS (via PuLP / Highs) com suporte a limite de tempo (time_limit).
5. Reconstrução determinística da rota ordenada da playlist.

Requer: pip install "pulp<4" highspy   (o PuLP 4.x mudou a API e não funciona com este código)
"""

import time
import numpy as np
import pulp
from typing import Dict, Any, List, Optional


def solve_open_atsp(
    cost_matrix: np.ndarray,
    time_limit: Optional[int] = 60,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Resolve o Open ATSP para uma matriz de custos n x n.
    
    cost_matrix: Matriz n x n contendo os custos c_ij entre faixas reais (0 a n-1).
    time_limit: Limite máximo de processamento em segundos (padrão 60s).
    verbose: Se True, imprime os logs internos do solver HiGHS.
    
    Retorna dicionário contendo:
    - 'status': Status de otimalidade ('Optimal', 'Not Solved', etc.)
    - 'objective_cost': Custo total ótimo da playlist
    - 'route': Lista de índices das faixas na ordem ótima [s_1, s_2, ..., s_n]
    - 'runtime_sec': Tempo de CPU em segundos
    - 'is_optimal': True somente se o solver PROVOU o ótimo. Se o tempo acabou e ele só
                    achou uma solução viável, é False (mesmo que 'status' diga 'Optimal').
    """
    n = cost_matrix.shape[0]
    
    # Nós: 0 a n-1 são as músicas reais. O nó 'n' é o nó fictício (dummy).
    dummy_node = n
    all_nodes = list(range(n + 1))
    real_nodes = list(range(n))
    
    # Cria o problema de minimização no PuLP
    prob = pulp.LpProblem("Open_ATSP_Playlist", pulp.LpMinimize)
    
    # Variáveis binárias de transição: x[(i, j)] = 1 se j toca após i
    valid_arcs = [(i, j) for i in all_nodes for j in all_nodes if i != j]
    x = pulp.LpVariable.dicts("x", valid_arcs, cat=pulp.LpBinary)
    
    # Variáveis contínuas de ordem para eliminação de sub-rotas (MTZ) sobre nós reais
    u = pulp.LpVariable.dicts("u", real_nodes, lowBound=1, upBound=n, cat=pulp.LpContinuous)
    
    # Função Objetivo: soma apenas os custos entre músicas reais (arcos com dummy custam 0)
    prob += pulp.lpSum(cost_matrix[i, j] * x[(i, j)] for i in real_nodes for j in real_nodes if i != j)
    
    # Restrição 1: Grau de Saída Estrito (cada nó possui exatamente um sucessor)
    for i in all_nodes:
        prob += pulp.lpSum(x[(i, j)] for j in all_nodes if j != i) == 1, f"OutDeg_{i}"
        
    # Restrição 2: Grau de Entrada Estrito (cada nó possui exatamente um antecessor)
    for j in all_nodes:
        prob += pulp.lpSum(x[(i, j)] for i in all_nodes if i != j) == 1, f"InDeg_{j}"
        
    # Restrição 3: Eliminação de Sub-rotas de Miller-Tucker-Zemlin (MTZ)
    # Impede a formação de ciclos isolados desconexos entre as músicas reais
    for i in real_nodes:
        for j in real_nodes:
            if i != j:
                prob += u[i] - u[j] + n * x[(i, j)] <= n - 1, f"MTZ_{i}_{j}"
                
    # Configura e executa o solver HiGHS
    solver = pulp.HiGHS(msg=verbose, timeLimit=time_limit)
    
    t_start = time.time()
    prob.solve(solver)
    runtime = time.time() - t_start
    
    # Distingue claramente: Ótimo provado vs Solução viável sob limite de tempo
    is_optimal = (prob.sol_status == pulp.constants.LpSolutionOptimal)
    has_solution = prob.sol_status in (pulp.constants.LpSolutionOptimal,
                                       pulp.constants.LpSolutionIntegerFeasible)
    
    if is_optimal:
        status_str = "Optimal"
    elif has_solution:
        status_str = "Time_Limit_Feasible"
    else:
        status_str = pulp.LpStatus[prob.status]
    
    # Reconstrução da rota da playlist:
    # O nó dummy 'n' aponta para a 1ª música real da playlist (x[dummy, s_1] == 1)
    # E a última música real aponta de volta para o dummy (x[s_n, dummy] == 1)
    route = []
    
    # Encontra o sucessor do nó dummy
    current = None
    for j in real_nodes:
        val = pulp.value(x[(dummy_node, j)])
        if val is not None and val > 0.5:
            current = j
            break
            
    if current is not None:
        route.append(current)
        # Segue a cadeia de sucessores até atingir todas as n músicas
        for _ in range(n - 1):
            next_node = None
            for j in real_nodes:
                if j != current:
                    val = pulp.value(x[(current, j)])
                    if val is not None and val > 0.5:
                        next_node = j
                        break
            if next_node is not None:
                route.append(next_node)
                current = next_node
            else:
                break
                
    # Validação estrita: a rota precisa conter todas as n faixas sem repetição
    is_valid_route = (len(route) == n and len(set(route)) == n)
    
    # Se o solver não achou solução ou a rota for incompleta, custo é nan
    if has_solution and is_valid_route:
        obj_val = float(pulp.value(prob.objective))
    else:
        obj_val = float('nan')
        route = []
        if not is_valid_route and has_solution:
            status_str = "Incomplete_Route"
            
    # Extração de métricas nativas do solver HiGHS
    solver_gap = 0.0 if is_optimal else None
    dual_bound = obj_val if is_optimal else None
    
    if hasattr(prob, 'solverModel') and prob.solverModel is not None:
        try:
            info = prob.solverModel.getInfo()
            if hasattr(info, 'mip_gap') and np.isfinite(info.mip_gap):
                if not is_optimal:
                    solver_gap = round(float(info.mip_gap) * 100.0, 2)
            if hasattr(info, 'mip_dual_bound') and np.isfinite(info.mip_dual_bound):
                if not is_optimal:
                    dual_bound = round(float(info.mip_dual_bound), 4)
        except Exception:
            pass
    
    return {
        'status': status_str,
        'objective_cost': obj_val,
        'route': route,
        'runtime_sec': runtime,
        'is_optimal': is_optimal,
        'solver_gap': solver_gap,
        'dual_bound': dual_bound,
        'num_tracks': n
    }
