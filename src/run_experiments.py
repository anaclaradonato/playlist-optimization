"""
Script de Bateria Experimental e Estudo Multiobjetivo (run_experiments.py)
Projeto: Sequenciamento Ótimo de Playlists (Open ATSP) - Trabalho Final de PL

Responsabilidades:
1. Executar bateria de testes com instâncias reais de diversos tamanhos:
   n = [30, 36, 40, 50, 60, 75, 100] faixas.
2. Comparar em cada instância:
   - Ordem Original do Spotify (Baseline Real);
   - Random Shuffle (Linha de base aleatória média);
   - Heurística Rápida (Vizinho Mais Próximo + Or-opt);
   - Solver Exato HiGHS (Branch-and-Cut com MTZ).
3. Calcular métricas formais de PO:
   - Tempo de CPU (segundos);
   - GAP percentual da heurística vs. ótimo global;
   - Taxa de redução de atrito acústico (%).
4. Salvar tabela consolidada em CSV ('tabela_resultados_experimentos.csv') e Markdown.
5. Gerar gráficos visuais em alta resolução para o artigo do SBPO:
   - 'grafico_evolucao_bpm.png': Perfil temporal de BPM (Antes vs. Depois);
   - 'grafico_pareto_multiobjetivo.png': Fronteira de Pareto entre Ritmo (BPM) e Harmonia (Tom).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend para salvar imagens sem abrir janelas de interface
import matplotlib.pyplot as plt

from data_loader import load_clean_dataset, get_playlist_by_id, list_sample_playlists
from cost_matrix import build_cost_matrix, evaluate_sequence_cost, get_sequence_breakdown
from solver import solve_open_atsp
from heuristics import random_shuffle_baseline, solve_heuristic


def get_results_dir() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "..", "results"),
        os.path.join(base, "results"),
        "results",
        base
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    target = os.path.abspath(os.path.join(base, "..", "results"))
    os.makedirs(target, exist_ok=True)
    return target


def run_multi_instance_benchmark(df_all: pd.DataFrame, target_sizes=[30, 36, 40, 50, 60, 75, 100]):
    """
    Executa a bateria de testes comparando Original, Shuffle, Heurística e HiGHS.
    """
    print("\n" + "=" * 85)
    print("       BATERIA DE EXPERIMENTOS MULTI-INSTÂNCIAS (30 a 100 MÚSICAS)")
    print("=" * 85)
    
    # Amostra playlists reais de diferentes tamanhos
    samples = list_sample_playlists(df_all, target_sizes=target_sizes)
    
    results = []
    
    for idx, row in samples.iterrows():
        pl_id = row['playlist_id']
        pl_name = row['playlist_name']
        pl_genre = row['playlist_genre']
        n_tracks = row['num_tracks']
        
        print(f"\n-> Processando Instância {idx+1}/{len(samples)}: '{pl_name}' ({pl_genre}) com n = {n_tracks} músicas...")
        
        df_pl = get_playlist_by_id(df_all, pl_id)
        
        # 1. Matriz de Custos Padrão
        c_matrix = build_cost_matrix(df_pl, w_bpm=0.35, w_tom=0.35, w_energia=0.30, gamma=1.5)
        
        # 2. Ordem Original do Spotify
        orig_seq = list(range(n_tracks))
        orig_cost = evaluate_sequence_cost(orig_seq, c_matrix)
        
        # 3. Linha de Base Aleatória (500 Shuffles)
        shuf_res = random_shuffle_baseline(c_matrix, n_shuffles=300)
        shuf_mean = shuf_res['mean_cost']
        
        # 4. Heurística (NN + Or-opt)
        heur_res = solve_heuristic(c_matrix)
        heur_cost = heur_res['cost']
        heur_time = heur_res['runtime_sec']
        
        # 5. Solver Exato HiGHS (com limite de 60s)
        time_limit = 60
        milp_res = solve_open_atsp(c_matrix, time_limit=time_limit, verbose=False)
        milp_cost = milp_res['objective_cost']
        milp_time = milp_res['runtime_sec']
        milp_status = milp_res['status']
        is_opt = milp_res['is_optimal']
        
        # GAP da Heurística em relação ao HiGHS (%)
        if milp_cost > 0:
            heur_gap = ((heur_cost - milp_cost) / milp_cost) * 100.0
        else:
            heur_gap = 0.0
            
        # Redução de Atrito do HiGHS em relação ao Original (%)
        improvement = ((orig_cost - milp_cost) / orig_cost) * 100.0 if orig_cost > 0 else 0.0
        
        results.append({
            'Tamanho_n': n_tracks,
            'Genero': pl_genre,
            'Nome_Playlist': pl_name[:24],
            'Custo_Original': round(orig_cost, 4),
            'Custo_Shuffle': round(shuf_mean, 4),
            'Tempo_Heur_s': round(heur_time, 4),
            'Custo_Heur': round(heur_cost, 4),
            'Tempo_HiGHS_s': round(milp_time, 3),
            'Custo_HiGHS': round(milp_cost, 4),
            'Status_HiGHS': milp_status,
            'GAP_Heur_%': round(heur_gap, 2),
            'Melhoria_%': round(improvement, 2)
        })
        
        print(f"   [OK] HiGHS: {milp_cost:.4f} ({milp_time:.2f}s) | Heurística: {heur_cost:.4f} (GAP: {heur_gap:.1f}%) | Redução: {improvement:.1f}%")
        
    df_results = pd.DataFrame(results)
    
    # Salva em CSV e Markdown na pasta de resultados
    res_dir = get_results_dir()
    csv_path = os.path.join(res_dir, "tabela_resultados_experimentos.csv")
    md_path = os.path.join(res_dir, "tabela_resultados_experimentos.md")
    
    df_results.to_csv(csv_path, index=False)
    try:
        table_md = df_results.to_markdown(index=False)
    except Exception:
        table_md = df_results.to_string(index=False)
        
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Resultados da Bateria Experimental (HiGHS vs. Heurística vs. Baseline)\n\n")
        f.write(table_md)
        
    print(f"\n-> Tabela salva com sucesso em:")
    print(f"   • CSV: {csv_path}")
    print(f"   • Markdown: {md_path}")
    
    return df_results


def generate_bpm_timeline_plot(df_all: pd.DataFrame, playlist_id="0DdGqNa18DwYyfIR05OrW1"):
    """
    Gera o gráfico comparativo do perfil de BPM (Original vs. HiGHS)
    demonstrando a suavização da curva rítmica.
    """
    df_pl = get_playlist_by_id(df_all, playlist_id)
    n = len(df_pl)
    
    c_matrix = build_cost_matrix(df_pl, w_bpm=0.35, w_tom=0.35, w_energia=0.30, gamma=1.5)
    milp_res = solve_open_atsp(c_matrix, time_limit=60, verbose=False)
    opt_route = milp_res['route']
    
    tempos_original = df_pl['tempo'].values
    tempos_otimizado = df_pl['tempo'].iloc[opt_route].values
    
    plt.figure(figsize=(10, 4.8), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    x = range(1, n + 1)
    
    plt.plot(x, tempos_original, marker='o', markersize=4, color='#94a3b8', linestyle='--', linewidth=1.5, label='Ordem Original (Spotify) — Saltos Caóticos')
    plt.plot(x, tempos_otimizado, marker='s', markersize=4, color='#0284c7', linewidth=2.2, label='Ordem Otimizada (HiGHS) — Transição Suave')
    
    plt.title(f"Perfil de Andamento (BPM) ao Longo da Playlist: Original vs. Otimizado (n = {n} faixas)", fontsize=11, fontweight='bold', pad=12, color='#0f172a')
    plt.xlabel("Posição de Execução da Música na Playlist", fontsize=9.5, fontweight='bold', color='#334155')
    plt.ylabel("Andamento (BPM)", fontsize=9.5, fontweight='bold', color='#334155')
    plt.xticks(range(1, n + 1, 2), fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9, loc='upper left')
    
    output_path = os.path.join(get_results_dir(), "grafico_evolucao_bpm.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"-> Gráfico de BPM salvo em: {output_path}")


def generate_pareto_frontier_plot(df_all: pd.DataFrame, playlist_id="0DdGqNa18DwYyfIR05OrW1"):
    """
    Estudo Multiobjetivo (Ponto Extra): Varredura paramétrica de pesos para gerar a Fronteira de Pareto
    demonstrando o trade-off entre Andamento Rítmico (BPM) e Harmonia Tonal (Camelot).
    """
    print("\n-> Executando Varredura da Fronteira de Pareto (Trade-off Ritmo vs. Harmonia)...")
    df_pl = get_playlist_by_id(df_all, playlist_id)
    n = len(df_pl)
    
    # Fixa o peso de energia em 0.20 e varia w_bpm de 0.05 a 0.75
    w_ene = 0.20
    bpm_weights = np.linspace(0.05, 0.75, 9)
    
    pareto_points = []
    
    for w_b in bpm_weights:
        w_t = round(1.0 - w_b - w_ene, 4)
        c_matrix = build_cost_matrix(df_pl, w_bpm=w_b, w_tom=w_t, w_energia=w_ene, gamma=1.5)
        
        # Resolve no HiGHS
        res = solve_open_atsp(c_matrix, time_limit=30, verbose=False)
        route = res['route']
        
        breakdown = get_sequence_breakdown(route, df_pl, gamma=1.5)
        
        pareto_points.append({
            'w_bpm': round(w_b, 2),
            'w_tom': round(w_t, 2),
            'avg_bpm_jump': breakdown['avg_bpm_jump'],
            'avg_camelot_dist': breakdown['avg_camelot_dist'],
            'harmonic_trans': breakdown['num_harmonic_transitions']
        })
        print(f"   • Peso BPM={w_b:.2f}, Tom={w_t:.2f} -> Salto BPM: {breakdown['avg_bpm_jump']:.1f} BPM | Dist. Camelot: {breakdown['avg_camelot_dist']:.2f}")
        
    df_pareto = pd.DataFrame(pareto_points)
    
    # Plota a curva da Fronteira de Pareto
    plt.figure(figsize=(8.5, 5), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    x_vals = df_pareto['avg_bpm_jump']
    y_vals = df_pareto['avg_camelot_dist']
    
    plt.plot(x_vals, y_vals, color='#0284c7', linestyle='-', linewidth=2, marker='o', markersize=6, label='Fronteira de Pareto Não-Dominada')
    
    # Anotações dos perfis
    for i, row in df_pareto.iterrows():
        plt.annotate(
            f"w_bpm={row['w_bpm']:.2f}",
            (row['avg_bpm_jump'], row['avg_camelot_dist']),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7.5,
            color='#1e293b'
        )
        
    plt.title("Fronteira de Pareto: Trade-off entre Salto de BPM e Distância Harmônica (Camelot)", fontsize=10.5, fontweight='bold', pad=12, color='#0f172a')
    plt.xlabel("Atrito Rítmico Médio (Salto Médio em BPM) → Menor é Melhor", fontsize=9, fontweight='bold', color='#334155')
    plt.ylabel("Atrito Harmônico Médio (Distância Camelot) → Menor é Melhor", fontsize=9, fontweight='bold', color='#334155')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5, loc='upper right')
    
    output_path = os.path.join(get_results_dir(), "grafico_pareto_multiobjetivo.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"-> Gráfico da Fronteira de Pareto salvo em: {output_path}")


def main():
    print("Iniciando Pipeline de Experimentos do Dia 2...")
    t0_all = time.time()
    
    # Carrega dados
    df_all = load_clean_dataset()
    
    # 1. Bateria Experimental Multi-instâncias (30 a 100 faixas)
    df_results = run_multi_instance_benchmark(df_all, target_sizes=[30, 36, 40, 50, 60, 75, 100])
    
    # 2. Gráfico de Evolução Temporal de BPM (Antes vs. Depois)
    generate_bpm_timeline_plot(df_all, playlist_id="0DdGqNa18DwYyfIR05OrW1")
    
    # 3. Gráfico da Fronteira de Pareto (Ponto Extra - Multiobjetivo)
    generate_pareto_frontier_plot(df_all, playlist_id="0DdGqNa18DwYyfIR05OrW1")
    
    total_sec = time.time() - t0_all
    print("\n" + "=" * 85)
    print(f"   TODOS OS EXPERIMENTOS DO DIA 2 FORAM CONCLUÍDOS EM {total_sec:.2f} SEGUNDOS!")
    print("=" * 85)


if __name__ == '__main__':
    main()

