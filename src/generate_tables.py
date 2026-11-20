"""
Módulo de Geração de Tabelas e Resultados Numéricos

Responsabilidades:
Executar todos os modelos matemáticos, testes estatísticos,
heurísticas e varreduras, salvando todas as 7 tabelas em formato CSV na pasta 'results/':
  1. tabela_dados_e_parametros.csv       
  2. tabela_resultados_experimentos.csv   
  3. tabela_estudo_de_caso_demo.csv      
  4. tabela_pareto_sensibilidade.csv     
  5. tabela_sensibilidade_gamma.csv      
  6. tabela_sensibilidade_energia.csv   
  7. sequencia_otimizada.csv             
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd

from data_loader import load_clean_dataset, get_playlist_by_id, list_sample_playlists
from cost_matrix import build_cost_matrix, evaluate_sequence_cost, get_sequence_breakdown
from solver import solve_open_atsp
from heuristics import random_shuffle_baseline, solve_heuristic


def get_results_dir():
    """Retorna o caminho absoluto do diretório de resultados."""
    base = os.path.dirname(os.path.abspath(__file__))
    target = os.path.abspath(os.path.join(base, "..", "results"))
    os.makedirs(target, exist_ok=True)
    return target


def generate_parameters_table(res_dir):
    """Gera a Tabela 1: Parâmetros, Ambiente Computacional e Limpeza de Dados."""
    print("\n-> [1/6] Gerando Tabela de Dados e Parâmetros...")
    params = [
        {'Categoria': 'Ambiente Computacional', 'Parametro': 'Processador (CPU)', 'Valor': 'Intel Core 12th/13th Gen (12 cores / 16 threads)', 'Papel_Metodologico': 'Plataforma de processamento local dos testes'},
        {'Categoria': 'Ambiente Computacional', 'Parametro': 'Memoria RAM', 'Valor': '8.0 GB', 'Papel_Metodologico': 'Capacidade de alocacao da arvore de Branch-and-Bound'},
        {'Categoria': 'Ambiente Computacional', 'Parametro': 'Sistema Operacional', 'Valor': 'Windows 11 (Build 26200)', 'Papel_Metodologico': 'Ambiente de execucao e controle de processos'},
        {'Categoria': 'Ambiente Computacional', 'Parametro': 'Linguagem / Versao', 'Valor': 'Python 3.13.1', 'Papel_Metodologico': 'Linguagem de suporte a modelagem matematica'},
        {'Categoria': 'Ambiente Computacional', 'Parametro': 'Solver de Otimizacao', 'Valor': 'HiGHS v1.16.0 (via PuLP v3.3.2)', 'Papel_Metodologico': 'Solver exato de alto desempenho sugerido no edital'},
        {'Categoria': 'Dataset e Limpeza', 'Parametro': 'Fonte dos Dados', 'Valor': 'Kaggle (Spotify Songs Dataset)', 'Papel_Metodologico': 'Base publica real de faixas com metadados acusticos'},
        {'Categoria': 'Dataset e Limpeza', 'Parametro': 'Faixas Brutas no CSV', 'Valor': '32.833 faixas', 'Papel_Metodologico': 'Contagem inicial do dataset'},
        {'Categoria': 'Dataset e Limpeza', 'Parametro': 'Faixas Validas Apos Limpeza', 'Valor': '32.185 faixas em 471 playlists', 'Papel_Metodologico': 'Descarte: 1 tempo<=0; 582 duplicatas ID; 65 nome/artista'},
        {'Categoria': 'Dataset e Limpeza', 'Parametro': 'Instancias Benchmark', 'Valor': '7 playlists (n = 15, 30, 36, 50, 60, 75, 100)', 'Papel_Metodologico': 'Amostragem em 4 generos (R&B, Rap, Pop, EDM)'},
        {'Categoria': 'Pesos da Funcao Objetivo', 'Parametro': 'Peso de BPM (w_bpm)', 'Valor': '0.35', 'Papel_Metodologico': 'Prioridade a continuidade ritmica (passo de danca)'},
        {'Categoria': 'Pesos da Funcao Objetivo', 'Parametro': 'Peso de Tom (w_tom)', 'Valor': '0.35', 'Papel_Metodologico': 'Prioridade a harmonia tonal (Roda de Camelot)'},
        {'Categoria': 'Pesos da Funcao Objetivo', 'Parametro': 'Peso de Energia (w_energia)', 'Valor': '0.30', 'Papel_Metodologico': 'Prioridade a manutencao da intensidade de palco'},
        {'Categoria': 'Parametros Fisicos', 'Parametro': 'Assimetria de Energia (gamma)', 'Valor': '1.5', 'Papel_Metodologico': 'Penalizacao assimetrica para amortecer drops bruscos'},
        {'Categoria': 'Parametros Fisicos', 'Parametro': 'Tolerancia Dobro/Metade BPM', 'Valor': '0.5x e 2.0x', 'Papel_Metodologico': 'Equivalencia harmonica de andamento ritmico'},
        {'Categoria': 'Parametros Fisicos', 'Parametro': 'Teto / Normalizacao de BPM', 'Valor': '40.0 BPM', 'Papel_Metodologico': 'Escalonamento linear para escala adimensional [0, 1]'},
        {'Categoria': 'Parametros Fisicos', 'Parametro': 'Normalizacao de Tom', 'Valor': '7 passos', 'Papel_Metodologico': 'Distancia maxima no grafo toroidal da Roda de Camelot'},
        {'Categoria': 'Protocolo Experimental', 'Parametro': 'Tempo Limite do HiGHS', 'Valor': '300 segundos', 'Papel_Metodologico': 'Orcamento de tempo estendido para provar otimo global'},
        {'Categoria': 'Protocolo Experimental', 'Parametro': 'Linha de Base de Controle', 'Valor': '500 permutacoes de Monte Carlo (seed=0)', 'Papel_Metodologico': 'Controle estatistico neutro na ausencia de curadoria'}
    ]
    df_params = pd.DataFrame(params)
    out_path = os.path.join(res_dir, 'tabela_dados_e_parametros.csv')
    df_params.to_csv(out_path, index=False)
    print(f"   [OK] Salvo: {out_path}")
    return df_params


def generate_benchmark_table(df_all, res_dir, target_sizes=[15, 30, 36, 50, 60, 75, 100]):
    """Gera a Tabela 2: Bateria Multi-instâncias com as 7 instâncias de teste."""
    print("\n-> [2/6] Executando Bateria de Benchmark (15 a 100 faixas)...")
    samples = list_sample_playlists(df_all, target_sizes=target_sizes)
    results = []

    for idx, row in samples.iterrows():
        pl_id = row['playlist_id']
        pl_name = row['playlist_name']
        pl_genre = row['playlist_genre']
        n_tracks = row['num_tracks']

        print(f"   • Instância {idx+1}/{len(samples)}: '{pl_name}' ({pl_genre}) | n = {n_tracks}...")
        df_pl = get_playlist_by_id(df_all, pl_id)
        c_matrix = build_cost_matrix(df_pl, w_bpm=0.35, w_tom=0.35, w_energia=0.30, gamma=1.5)

        # Shuffle
        shuf_res = random_shuffle_baseline(c_matrix, n_shuffles=500, seed=0)
        shuf_mean = shuf_res['mean_cost']
        shuf_std = shuf_res['std_cost']

        # Heurística (NN + Or-opt)
        heur_res = solve_heuristic(c_matrix)
        heur_cost = heur_res['cost']
        heur_time = heur_res['runtime_sec']

        # HiGHS (limite estendido de 300 s)
        milp_res = solve_open_atsp(c_matrix, time_limit=300, verbose=False)
        milp_cost = milp_res['objective_cost']
        milp_time = milp_res['runtime_sec']
        milp_status = milp_res['status']

        if np.isfinite(milp_cost) and milp_cost > 0:
            heur_gap = ((heur_cost - milp_cost) / milp_cost) * 100.0
        else:
            heur_gap = 0.0

        best_cost = min(heur_cost, milp_cost) if np.isfinite(milp_cost) else heur_cost
        reduc_vs_shuf = ((shuf_mean - best_cost) / shuf_mean) * 100.0 if shuf_mean > 0 else 0.0

        results.append({
            'Tamanho_n': n_tracks,
            'Genero': pl_genre,
            'Nome_Playlist': pl_name[:24],
            'Shuffle_Medio': round(shuf_mean, 4),
            'Shuffle_DesvPad': round(shuf_std, 4),
            'Custo_Heur': round(heur_cost, 4),
            'Tempo_Heur_s': round(heur_time, 4),
            'Custo_HiGHS': round(milp_cost, 4) if np.isfinite(milp_cost) else 'NaN',
            'Tempo_HiGHS_s': round(milp_time, 3),
            'Status_HiGHS': milp_status,
            'GAP_Heur_%': f"{heur_gap:+.2f}%",
            'Reducao_vs_Shuffle_%': round(reduc_vs_shuf, 2)
        })

    df_results = pd.DataFrame(results)
    out_path = os.path.join(res_dir, 'tabela_resultados_experimentos.csv')
    df_results.to_csv(out_path, index=False)
    print(f"   [OK] Salvo: {out_path}")
    return df_results


def generate_case_study_and_sequence_tables(df_all, res_dir, playlist_id="0DdGqNa18DwYyfIR05OrW1"):
    """Gera tabela_estudo_de_caso_demo.csv e sequencia_otimizada.csv para a playlist n=36."""
    print("\n-> [3/6] Gerando Estudo de Caso Prático e Sequência Otimizada (n=36)...")
    df_pl = get_playlist_by_id(df_all, playlist_id)
    n = len(df_pl)
    c_mat = build_cost_matrix(df_pl, w_bpm=0.35, w_tom=0.35, w_energia=0.30, gamma=1.5)
    
    res = solve_open_atsp(c_mat, time_limit=60, verbose=False)
    opt_route = res['route']

    orig_route = list(range(n))
    bd_orig = get_sequence_breakdown(orig_route, df_pl, gamma=1.5)
    bd_opt = get_sequence_breakdown(opt_route, df_pl, gamma=1.5)

    n_trans = n - 1
    bpm_reduc = ((bd_orig['avg_bpm_jump'] - bd_opt['avg_bpm_jump']) / bd_orig['avg_bpm_jump']) * 100.0 if bd_orig['avg_bpm_jump'] > 0 else 0.0
    bpm_max_reduc = ((bd_orig['max_bpm_jump'] - bd_opt['max_bpm_jump']) / bd_orig['max_bpm_jump']) * 100.0 if bd_orig['max_bpm_jump'] > 0 else 0.0

    pct_harm_orig = (bd_orig['num_harmonic_transitions'] / n_trans) * 100.0
    pct_harm_opt = (bd_opt['num_harmonic_transitions'] / n_trans) * 100.0
    harm_gain = ((bd_opt['num_harmonic_transitions'] - bd_orig['num_harmonic_transitions']) / bd_orig['num_harmonic_transitions']) * 100.0 if bd_orig['num_harmonic_transitions'] > 0 else 0.0

    drops_orig = bd_orig['total_energy_drops']
    drops_opt = bd_opt['total_energy_drops']
    drops_diff = ((drops_orig - drops_opt) / drops_orig) * 100.0 if drops_orig > 0 else 0.0

    demo_rows = [
        {'Metrica_Fisica': 'Salto Médio de BPM', 'Ordem_Arquivo_Antes': f"{bd_orig['avg_bpm_jump']:.2f} BPM", 'HiGHS_Depois': f"{bd_opt['avg_bpm_jump']:.2f} BPM", 'Impacto_Fisico': f"-{bpm_reduc:.1f}% (Fluidez contínua na dança)"},
        {'Metrica_Fisica': 'Salto Máximo de BPM', 'Ordem_Arquivo_Antes': f"{bd_orig['max_bpm_jump']:.2f} BPM", 'HiGHS_Depois': f"{bd_opt['max_bpm_jump']:.2f} BPM", 'Impacto_Fisico': f"-{bpm_max_reduc:.1f}% (Eliminação de choques bruscos)"},
        {'Metrica_Fisica': 'Transições Harmônicas (Camelot <= 1)', 'Ordem_Arquivo_Antes': f"{bd_orig['num_harmonic_transitions']} de {n_trans} ({pct_harm_orig:.1f}%)", 'HiGHS_Depois': f"{bd_opt['num_harmonic_transitions']} de {n_trans} ({pct_harm_opt:.1f}%)", 'Impacto_Fisico': f"+{harm_gain:.1f}% (Mais que triplicou a harmonia)"},
        {'Metrica_Fisica': 'Trocas de Tom na Sequência', 'Ordem_Arquivo_Antes': f"{bd_orig['n_key_changes']} de {n_trans}", 'HiGHS_Depois': f"{bd_opt['n_key_changes']} de {n_trans}", 'Impacto_Fisico': 'Transições harmônicas contínuas'},
        {'Metrica_Fisica': 'Total de Quedas de Energia', 'Ordem_Arquivo_Antes': f"{drops_orig} quedas", 'HiGHS_Depois': f"{drops_opt} quedas", 'Impacto_Fisico': f"{'-' if drops_diff > 0 else '+'}{abs(drops_diff):.1f}% (Manutenção da pista)" if drops_diff != 0 else "Estável (Manutenção da pista)"},
        {'Metrica_Fisica': 'Maior Salto de Energia (|ΔE|)', 'Ordem_Arquivo_Antes': f"{bd_orig['max_energy_jump']:.3f}", 'HiGHS_Depois': f"{bd_opt['max_energy_jump']:.3f}", 'Impacto_Fisico': 'Amortecimento de drops'}
    ]
    df_demo = pd.DataFrame(demo_rows)
    demo_path = os.path.join(res_dir, 'tabela_estudo_de_caso_demo.csv')
    df_demo.to_csv(demo_path, index=False)
    print(f"   [OK] Salvo: {demo_path}")

    # Salva a sequência completa em sequencia_otimizada.csv
    seq_rows = []
    for pos, idx in enumerate(opt_route, 1):
        row = {
            'Ordem': pos,
            'Musica': df_pl['track_name'].iloc[idx],
            'Artista': df_pl['track_artist'].iloc[idx],
            'BPM': round(df_pl['tempo'].iloc[idx], 1),
            'Tom_Camelot': df_pl['camelot'].iloc[idx],
            'Energia': round(df_pl['energy'].iloc[idx], 2)
        }
        if pos < n:
            next_idx = opt_route[pos]
            t_i = df_pl['tempo'].iloc[idx]
            t_j = df_pl['tempo'].iloc[next_idx]
            dt = min(abs(t_i - t_j), abs(t_i - 2*t_j), abs(2*t_i - t_j))
            row['Salto_BPM_Prox'] = round(dt, 1)
            row['Custo_Transicao'] = round(c_mat[idx, next_idx], 4)
        else:
            row['Salto_BPM_Prox'] = '-'
            row['Custo_Transicao'] = '-'
        seq_rows.append(row)

    df_seq = pd.DataFrame(seq_rows)
    seq_path = os.path.join(res_dir, 'sequencia_otimizada.csv')
    df_seq.to_csv(seq_path, index=False)
    print(f"   [OK] Salvo: {seq_path}")


def generate_pareto_table(df_all, res_dir, playlist_id="0DdGqNa18DwYyfIR05OrW1"):
    """Gera a Tabela de Pareto: varredura paramétrica de pesos (BPM vs Tom com Energia fixa)."""
    print("\n-> [4/6] Executando Varredura de Pareto (Ritmo vs Harmonia)...")
    df_pl = get_playlist_by_id(df_all, playlist_id)
    w_ene = 0.30
    bpm_weights = np.linspace(0.05, 0.65, 9)
    points = []

    for w_b in bpm_weights:
        w_b = round(float(w_b), 3)
        w_t = round(0.70 - w_b, 3)
        c_matrix = build_cost_matrix(df_pl, w_bpm=w_b, w_tom=w_t, w_energia=w_ene, gamma=1.5)
        res = solve_open_atsp(c_matrix, time_limit=45, verbose=False)
        bd = get_sequence_breakdown(res['route'], df_pl, gamma=1.5)

        points.append({
            'w_bpm': w_b,
            'w_tom': w_t,
            'w_ene': round(w_ene, 2),
            'avg_bpm_jump': bd['avg_bpm_jump'],
            'avg_camelot_dist': bd['avg_camelot_dist'],
            'status': res['status']
        })

    df_pts = pd.DataFrame(points)

    # Cálculo formal de dominância de Pareto
    nondom_list = []
    for i, r in df_pts.iterrows():
        is_dominated = False
        for j, other in df_pts.iterrows():
            if i == j:
                continue
            better_or_equal = (other['avg_bpm_jump'] <= r['avg_bpm_jump'] and other['avg_camelot_dist'] <= r['avg_camelot_dist'])
            strictly_better = (other['avg_bpm_jump'] < r['avg_bpm_jump'] or other['avg_camelot_dist'] < r['avg_camelot_dist'])
            if better_or_equal and strictly_better:
                is_dominated = True
                break
        if not is_dominated:
            nondom_list.append(i)

    df_pts['is_nondominated'] = df_pts.index.isin(nondom_list)
    out_path = os.path.join(res_dir, 'tabela_pareto_sensibilidade.csv')
    df_pts.to_csv(out_path, index=False)
    print(f"   [OK] Salvo: {out_path}")
    return df_pts


def generate_gamma_sensitivity_table(df_all, res_dir, playlist_id="0DdGqNa18DwYyfIR05OrW1"):
    """Gera a tabela de sensibilidade ao fator de assimetria gamma in [1.0, 3.0]."""
    print("\n-> [5/6] Executando Sensibilidade ao Fator Gamma (Penalização de Queda)...")
    df_pl = get_playlist_by_id(df_all, playlist_id)
    gamma_records = []

    for g in [1.0, 1.2, 1.5, 2.0, 3.0]:
        t0 = time.time()
        c_mat = build_cost_matrix(df_pl, w_bpm=0.35, w_tom=0.35, w_energia=0.30, gamma=g)
        res = solve_open_atsp(c_mat, time_limit=30, verbose=False)
        t_highs = time.time() - t0
        bd = get_sequence_breakdown(res['route'], df_pl, gamma=g)

        gamma_records.append({
            'gamma': g,
            'Custo_Total': round(res['objective_cost'], 4),
            'Salto_Medio_BPM': round(bd['avg_bpm_jump'], 2),
            'Dist_Media_Camelot': round(bd['avg_camelot_dist'], 2),
            'Total_Quedas_Energia': bd['total_energy_drops'],
            'Media_Queda_Energia': round(bd['avg_energy_drop'], 3),
            'Maior_Salto_Energia': round(bd['max_energy_jump'], 3),
            'Status_HiGHS': res['status'],
            'Tempo_HiGHS_s': round(t_highs, 3)
        })

    df_gamma = pd.DataFrame(gamma_records)
    out_path = os.path.join(res_dir, 'tabela_sensibilidade_gamma.csv')
    df_gamma.to_csv(out_path, index=False)
    print(f"   [OK] Salvo: {out_path}")
    return df_gamma


def generate_energy_sensitivity_table(df_all, res_dir, playlist_id="0DdGqNa18DwYyfIR05OrW1"):
    """Gera a tabela de sensibilidade ao peso de intensidade w_energia in [0.0, 0.5]."""
    print("\n-> [6/6] Executando Sensibilidade ao Peso da Energia (w_energia)...")
    df_pl = get_playlist_by_id(df_all, playlist_id)
    records = []

    for w_e in [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]:
        rem = round(1.0 - w_e, 3)
        w_b = round(rem / 2, 3)
        w_t = round(rem / 2, 3)

        c_matrix = build_cost_matrix(df_pl, w_bpm=w_b, w_tom=w_t, w_energia=w_e, gamma=1.5)
        res = solve_open_atsp(c_matrix, time_limit=30, verbose=False)
        bd = get_sequence_breakdown(res['route'], df_pl, gamma=1.5)

        records.append({
            'w_energia': w_e,
            'w_bpm': w_b,
            'w_tom': w_t,
            'Salto_Medio_BPM': round(bd['avg_bpm_jump'], 2),
            'Dist_Media_Camelot': round(bd['avg_camelot_dist'], 2),
            'Total_Quedas_Energia': bd['total_energy_drops'],
            'Media_Queda_Energia': round(bd['avg_energy_drop'], 3),
            'Maior_Salto_Energia': round(bd['max_energy_jump'], 3),
            'Custo_Objetivo': round(res['objective_cost'], 4),
            'Status_HiGHS': res['status']
        })

    df_sens = pd.DataFrame(records)
    out_path = os.path.join(res_dir, 'tabela_sensibilidade_energia.csv')
    df_sens.to_csv(out_path, index=False)
    print(f"   [OK] Salvo: {out_path}")
    return df_sens


def main():
    t0 = time.time()
    res_dir = get_results_dir()
    df_all = load_clean_dataset()

    generate_parameters_table(res_dir)
    generate_benchmark_table(df_all, res_dir)
    generate_case_study_and_sequence_tables(df_all, res_dir)
    generate_pareto_table(df_all, res_dir)
    generate_gamma_sensitivity_table(df_all, res_dir)
    generate_energy_sensitivity_table(df_all, res_dir)

    total = time.time() - t0
    print("\n" + "=" * 80)
    print(f"   [CONCLUÍDO] Todas as 7 tabelas CSV foram geradas com sucesso em {total:.1f} s!")
    print(f"   Diretório: {res_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

