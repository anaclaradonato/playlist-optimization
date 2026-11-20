"""
Script de Execução Principal

Objetivos:
1. Carregar os dados limpos do 'spotify_songs.csv'.
2. Selecionar uma playlist real com n = 36 músicas.
3. Construir a matriz de custos multicritério c_ij (BPM com dobro/metade, Camelot /7, Energia com gama).
4. Resolver o Open ATSP com o solver HiGHS (MTZ + nó fictício n).
5. Exibir métricas comparativas: Custo Original vs. Custo Ótimo, tempo de CPU e redução de atrito.
"""

import os
import sys
import time
import random
import pandas as pd
from data_loader import load_clean_dataset, get_playlist_by_id, list_sample_playlists
from cost_matrix import build_cost_matrix, evaluate_sequence_cost, get_sequence_breakdown
from solver import solve_open_atsp


def main():
    
    # Carrega o dataset limpo
    df = load_clean_dataset()
    
    # Suporte dinâmico a qualquer playlist:
    # 1. Pelo tamanho: ex. 'python src/main.py 40' ou 'python src/main.py 50'
    # 2. Pelo ID: ex. 'python src/main.py <playlist_id>'
    # 3. Padrão (sem argumentos): playlist real com 36 faixas ('Viral Southern Hip Hop')
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target_arg is None:
        playlist_id = "0DdGqNa18DwYyfIR05OrW1"
        df_pl = get_playlist_by_id(df, playlist_id)
    elif target_arg.isdigit():
        target_size = int(target_arg)
        sample = list_sample_playlists(df, target_sizes=[target_size])
        playlist_id = sample.iloc[0]['playlist_id']
        df_pl = get_playlist_by_id(df, playlist_id)
    else:
        playlist_id = target_arg
        df_pl = get_playlist_by_id(df, playlist_id)
        
    n = len(df_pl)
    pl_name = df_pl['playlist_name'].iloc[0]
    pl_genre = df_pl['playlist_genre'].iloc[0]
    
    print(f"\n[Playlist Selecionada para a Otimização]")
    print(f" • Nome:        '{pl_name}'")
    print(f" • ID:          {playlist_id}")
    print(f" • Gênero:      {pl_genre}")
    print(f" • Quantidade:  {n} músicas")
    
    # Diagnóstico da Heterogeneidade Acústica Intra-Gênero
    bpm_min = df_pl['tempo'].min()
    bpm_max = df_pl['tempo'].max()
    bpm_mean = df_pl['tempo'].mean()
    bpm_std = df_pl['tempo'].std()
    
    e_min = df_pl['energy'].min()
    e_max = df_pl['energy'].max()
    e_mean = df_pl['energy'].mean()
    e_std = df_pl['energy'].std()
    
    n_keys = df_pl['camelot'].nunique()
    
    print("\n[Diagnóstico de Heterogeneidade Acústica Intra-Gênero]")
    print(f" • BPM:     Variação de {bpm_min:.1f} a {bpm_max:.1f} BPM (Média: {bpm_mean:.1f} | Desvio-Padrão: {bpm_std:.1f})")
    print(f" • Energia: Variação de {e_min:.2f} a {e_max:.2f} (Média: {e_mean:.2f} | Desvio-Padrão: {e_std:.2f})")
    print(f" • Tom:     {n_keys} códigos distintos na Roda de Camelot")
    print(f" -> Conclusão: Alta dispersão física intra-gênero justifica a otimização matemática das transições!")
    
    # Parâmetros da Função de Custo
    w_bpm = 0.35
    w_tom = 0.35
    w_energia = 0.30
    gamma = 1.5
    
    print(f"\n[Parâmetros da Função de Custo c_ij]")
    print(f" • Peso BPM:     w_bpm = {w_bpm:.2f} (com dobro e metade do tempo)")
    print(f" • Peso Tom:     w_tom = {w_tom:.2f} (distância Roda de Camelot / 7)")
    print(f" • Peso Energia: w_energia = {w_energia:.2f} (assimetria γ = {gamma:.1f} para quedas)")
    
    c_matrix = build_cost_matrix(
        df_pl,
        w_bpm=w_bpm,
        w_tom=w_tom,
        w_energia=w_energia,
        gamma=gamma
    )
    print("   Matriz calculada com sucesso!")
    
    # Ordem do arquivo (a ordem em que as faixas aparecem no CSV, antes da otimização).
    original_seq = list(range(n))
    orig_cost = evaluate_sequence_cost(original_seq, c_matrix)
    orig_breakdown = get_sequence_breakdown(original_seq, df_pl, gamma=gamma)

    # Referência: custo médio de 500 ordens aleatórias (semente fixa 0 para repetir o resultado)
    random.seed(0)
    random_costs = []
    for _ in range(500):
        seq = list(range(n))
        random.shuffle(seq)
        random_costs.append(evaluate_sequence_cost(seq, c_matrix))
    random_mean = sum(random_costs) / len(random_costs)
    
    # Resolução com HiGHS
    print("\n-> Executando o solver HiGHS (Open ATSP com formulação MTZ)...")
    t0 = time.time()
    result = solve_open_atsp(c_matrix, time_limit=60, verbose=False)
    elapsed = time.time() - t0
    
    opt_route = result['route']
    opt_cost = result['objective_cost']
    status = result['status']
    is_optimal = result['is_optimal']
    
    # Se o solver não achou nenhuma solução dentro do limite de tempo
    if len(opt_route) == 0:
        print("[AVISO] O solver não encontrou nenhuma solução viável dentro do limite de tempo.")
        return

    # Validação da rota
    if len(opt_route) != n or len(set(opt_route)) != n:
        print("[ERRO] A rota gerada não contém todas as faixas exatamente uma vez!")
        return
        
    opt_breakdown = get_sequence_breakdown(opt_route, df_pl, gamma=gamma)
    
    # Relatório de Resultados
    print("\n" + "=" * 75)
    print("                    RESULTADOS OBTIDOS")
    print("=" * 75)
    print(f" • Status do Solver:                  {status} (Ótimo Global Comprovado: {is_optimal})")
    print(f" • Tempo de Execução:                 {result['runtime_sec']:.3f} segundos")
    print(f" • Custo da ordem do arquivo:         {orig_cost:.4f}")
    print(f" • Custo médio de ordens aleatórias:   {random_mean:.4f}  (500 sorteios)")
    if is_optimal:
        print(f" • Custo da ordem ÓTIMA:              {opt_cost:.4f}")
    else:
        print(f" • Custo da melhor ordem encontrada (ótimo não provado): {opt_cost:.4f}")
    
    improvement = ((orig_cost - opt_cost) / orig_cost) * 100.0 if orig_cost > 0 else 0.0
    improvement_random = ((random_mean - opt_cost) / random_mean) * 100.0 if random_mean > 0 else 0.0
    print(f" • Redução de custo vs. ordem do arquivo:    {improvement:.2f}%")
    print(f" • Redução de custo vs. média das aleatórias: {improvement_random:.2f}%")
    print("-" * 75)
    
    print("\n[Comparativo Físico das Transições]")
    print(f"{'Métrica Avaliada':<35} | {'Ordem do arquivo':<20} | {'Otimizado (HiGHS)':<20}")
    print("-" * 80)
    print(f"{'Salto Médio de BPM':<35} | {orig_breakdown['avg_bpm_jump']:>16.1f} BPM | {opt_breakdown['avg_bpm_jump']:>16.1f} BPM")
    print(f"{'Salto Máximo de BPM':<35} | {orig_breakdown['max_bpm_jump']:>16.1f} BPM | {opt_breakdown['max_bpm_jump']:>16.1f} BPM")
    print(f"{'Distância Média Camelot':<35} | {orig_breakdown['avg_camelot_dist']:>18.2f} | {opt_breakdown['avg_camelot_dist']:>18.2f}")
    print(f"{'Transições Harmônicas (dist <= 1)':<35} | {orig_breakdown['num_harmonic_transitions']:>14} de {n-1} | {opt_breakdown['num_harmonic_transitions']:>14} de {n-1}")
    print(f"{'Trocas de Tom (código Camelot)':<35} | {orig_breakdown['n_key_changes']:>14} de {n-1} | {opt_breakdown['n_key_changes']:>14} de {n-1}")
    print(f"{'Total de Quedas Bruscas de Energia':<35} | {orig_breakdown['total_energy_drops']:>18} | {opt_breakdown['total_energy_drops']:>18}")
    print(f"{'Soma dos Saltos de BPM':<35} | {orig_breakdown['total_bpm_jump']:>16.1f} BPM | {opt_breakdown['total_bpm_jump']:>16.1f} BPM")
    print(f"{'Maior Salto de Energia (|ΔE|)':<35} | {orig_breakdown['max_energy_jump']:>18.2f} | {opt_breakdown['max_energy_jump']:>18.2f}")
    print("-" * 80)
    
    print("\n[Exemplo das Primeiras 6 Faixas Sequenciadas pelo HiGHS]")
    for pos, track_idx in enumerate(opt_route[:6], 1):
        name = df_pl['track_name'].iloc[track_idx][:28]
        artist = df_pl['track_artist'].iloc[track_idx][:18]
        bpm = df_pl['tempo'].iloc[track_idx]
        cam = df_pl['camelot'].iloc[track_idx]
        ene = df_pl['energy'].iloc[track_idx]
        print(f" {pos:>2}º: {name:<28} | {artist:<18} | {bpm:>5.1f} BPM | Tom: {cam:>3} | E: {ene:.2f}")

    # Salva a sequência completa de todas as faixas em CSV
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.abspath(os.path.join(base_dir, "..", "results"))
    os.makedirs(res_dir, exist_ok=True)
    
    rows = []
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
            row['Custo_Transicao'] = round(c_matrix[idx, next_idx], 4)
        else:
            row['Salto_BPM_Prox'] = '-'
            row['Custo_Transicao'] = '-'
        rows.append(row)
        
    df_seq = pd.DataFrame(rows)
    
    out_csv = os.path.join(res_dir, "sequencia_otimizada.csv")
    df_seq.to_csv(out_csv, index=False)
        
    print(f"\n-> Sequência completa das {n} faixas salva com sucesso em:")
    print(f"   • CSV:      {out_csv}")


if __name__ == '__main__':
    main()

