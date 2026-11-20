"""
Módulo de Matriz de Custo de Transição

Responsabilidades:
1. Cálculo do custo rítmico (BPM) com tratamento de tempo duplo e metade (|a-b|, |a-2b|, |2a-b|).
2. Cálculo do custo harmônico na Roda de Camelot (distância circular horária + modulação modal / 7).
3. Cálculo do custo de energia acústica assimétrica com penalidade gama (γ) para quedas.
4. Montagem da matriz de adjacência c_ij (n x n) ponderada:
   c_ij = w_bpm * BPM_ij + w_tom * tom_ij + w_energia * energia_ij
5. Funções para avaliar o custo de uma sequência
"""

import numpy as np
import pandas as pd


def calc_bpm_cost(bpm_a, bpm_b, max_jump=40.0):
    """
    Calcula a descontinuidade de andamento considerando dobro e metade do tempo:
    delta_t = min(|a - b|, |a - 2b|, |2a - b|)
    Truncado em max_jump (padrão 40 BPM) e normalizado em [0, 1]
    """
    delta_t = min(
        abs(bpm_a - bpm_b),
        abs(bpm_a - 2.0 * bpm_b),
        abs(2.0 * bpm_a - bpm_b)
    )
    return min(delta_t, max_jump) / max_jump # normalização


def parse_camelot(camelot_str):
    """
    Separa o código Camelot em número de 1 a 12 e letra 'A' ou 'B'
    """
    camelot_str = str(camelot_str).strip().upper()
    letter = camelot_str[-1]
    number = int(camelot_str[:-1])
    return number, letter


def calc_camelot_cost(camelot_a, camelot_b):
    """
    Calcula a distância harmônica na Roda de Camelot dividida por 7 (escala 0 a 1):
    - Distância horária circular: min(|h1 - h2|, 12 - |h1 - h2|) (varia de 0 a 6)
    - Penalidade de troca de anel maior/menor: +1 se as letras diferem
    - Distância máxima teórica: 7
    """
    num_a, let_a = parse_camelot(camelot_a)
    num_b, let_b = parse_camelot(camelot_b)
    
    diff_h = abs(num_a - num_b)
    circular_dist = min(diff_h, 12 - diff_h)
    ring_penalty = 1 if let_a != let_b else 0
    
    total_dist = circular_dist + ring_penalty
    return total_dist / 7.0 # normalização


def calc_energy_cost(energy_a, energy_b, gamma=1.5):
    """
    Calcula o custo da mudança de energia acústica:
    - Se a energia sobe ou se mantém (E_b >= E_a): custo = |E_b - E_a|
    - Se a energia cai (E_b < E_a): custo = gamma * |E_b - E_a| (torna quedas de intensidade mais caras para o solver)
    Com gamma != 1, induz assimetria estrita no grafo (c_ij != c_ji)
    """
    delta_e = energy_b - energy_a
    if delta_e >= 0:
        return float(delta_e)
    else:
        return float(gamma * abs(delta_e))


def build_cost_matrix(
    df_playlist,
    w_bpm=0.35,
    w_tom=0.35,
    w_energia=0.30,
    gamma=1.5,
    max_bpm_jump=40.0
):
    """
    Constrói a matriz n x n de custos de transição c_ij para uma playlist de n músicas
    A diagonal principal (i == j) é preenchida com infinito para evitar auto-laços
    """
    n = len(df_playlist)
    c_matrix = np.zeros((n, n), dtype=float)
    
    tempos = df_playlist['tempo'].values
    camelots = df_playlist['camelot'].values
    energies = df_playlist['energy'].values
    
    for i in range(n):
        for j in range(n):
            if i == j:
                c_matrix[i, j] = 1e9 # Proíbe transição para si mesma
            else:
                cost_bpm = calc_bpm_cost(tempos[i], tempos[j], max_jump=max_bpm_jump)
                cost_tom = calc_camelot_cost(camelots[i], camelots[j])
                cost_ene = calc_energy_cost(energies[i], energies[j], gamma=gamma)
                
                c_matrix[i, j] = (
                    w_bpm * cost_bpm +
                    w_tom * cost_tom +
                    w_energia * cost_ene
                )
                
    return c_matrix


def evaluate_sequence_cost(sequence, cost_matrix):
    """
    Calcula o custo total agregado de uma sequência de faixas (caminho aberto)
    sequence: lista de índices ordenados das faixas
    """
    total_cost = 0.0
    for k in range(len(sequence) - 1):
        i = sequence[k]
        j = sequence[k + 1]
        total_cost += cost_matrix[i, j]
    return float(total_cost)


def get_sequence_breakdown(sequence, df_playlist, gamma=1.5):
    """
    Extrai métricas físicas detalhadas de uma sequência ordenada de faixas
    """
    n = len(sequence)
    tempos = df_playlist['tempo'].values
    camelots = df_playlist['camelot'].values
    energies = df_playlist['energy'].values
    
    bpm_jumps, camelot_dists, energy_drops, energy_jumps = [], [], [], []
    key_changes = 0
    
    for k in range(n - 1):
        i, j = sequence[k], sequence[k + 1]
        
        # Variação rítmica (BPM com equivalência de dobro/metade)
        dt = min(abs(tempos[i] - tempos[j]), abs(tempos[i] - 2.0 * tempos[j]), abs(2.0 * tempos[i] - tempos[j]))
        bpm_jumps.append(dt)
        
        # Distância harmônica na Roda de Camelot
        cd = int(round(calc_camelot_cost(camelots[i], camelots[j]) * 7.0))
        camelot_dists.append(cd)
        if cd > 0:
            key_changes += 1
            
        # Variação e quedas de energia
        de = energies[j] - energies[i]
        energy_jumps.append(abs(de))
        if de < 0:
            energy_drops.append(abs(de))
            
    return {
        'num_tracks': n,
        'avg_bpm_jump': float(np.mean(bpm_jumps)) if bpm_jumps else 0.0,
        'max_bpm_jump': float(np.max(bpm_jumps)) if bpm_jumps else 0.0,
        'avg_camelot_dist': float(np.mean(camelot_dists)) if camelot_dists else 0.0,
        'num_harmonic_transitions': sum(1 for d in camelot_dists if d <= 1),
        'total_energy_drops': len(energy_drops),
        'avg_energy_drop': float(np.mean(energy_drops)) if energy_drops else 0.0,
        'total_bpm_jump': float(np.sum(bpm_jumps)) if bpm_jumps else 0.0,
        'n_key_changes': key_changes,
        'max_energy_jump': float(np.max(energy_jumps)) if energy_jumps else 0.0
    }
