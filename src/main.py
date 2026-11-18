"""
Script de Execução Principal (main.py)
Projeto: Sequenciamento Ótimo de Playlists (Open ATSP) - Trabalho Final de PL

Objetivos:
1. Carregar os dados limpos do 'spotify_songs.csv'.
2. Selecionar uma playlist real com n = 36 músicas.
3. Construir a matriz de custos multicritério c_ij (BPM com dobro/metade, Camelot /7, Energia com gama).
4. Resolver o Open ATSP com o solver HiGHS (MTZ + nó fictício n).
5. Exibir métricas comparativas: Custo Original vs. Custo Ótimo, tempo de CPU e redução de atrito.
"""

import time
import pandas as pd
from data_loader import load_clean_dataset, get_playlist_by_id, list_sample_playlists
from cost_matrix import build_cost_matrix, evaluate_sequence_cost, get_sequence_breakdown
from solver import solve_open_atsp


def main():
    
    # Carrega o dataset limpo
    df = load_clean_dataset()
    
    # Escolhemos a playlist 'Viral Southern Hip Hop' com 36 faixas para rodar o exemplo
    playlist_id = "0DdGqNa18DwYyfIR05OrW1"
    df_pl = get_playlist_by_id(df, playlist_id)
    n = len(df_pl)
    
    pl_name = df_pl['playlist_name'].iloc[0]
    pl_genre = df_pl['playlist_genre'].iloc[0]
    
    print(f"\n[Playlist Selecionada para o Teste]")
    print(f" • Nome:   '{pl_name}'")
    print(f" • Gênero: {pl_genre}")
    print(f" • Faixas: {n} músicas")
    
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
    
    # Avaliação da Ordem Original (antes da otimização)
    original_seq = list(range(n))
    orig_cost = evaluate_sequence_cost(original_seq, c_matrix)
    orig_breakdown = get_sequence_breakdown(original_seq, df_pl, gamma=gamma)
    
    # Resolução com HiGHS
    print("\n-> Executando o solver HiGHS (Open ATSP com formulação MTZ)...")
    t0 = time.time()
    result = solve_open_atsp(c_matrix, time_limit=60, verbose=False)
    elapsed = time.time() - t0
    
    opt_route = result['route']
    opt_cost = result['objective_cost']
    status = result['status']
    is_optimal = result['is_optimal']
    
    # Validação da rota
    if len(opt_route) != n or len(set(opt_route)) != n:
        print("[ERRO] A rota gerada não contém todas as faixas exatamente uma vez!")
        return
        
    opt_breakdown = get_sequence_breakdown(opt_route, df_pl, gamma=gamma)
    
    # Relatório de Resultados
    print("\n" + "=" * 75)
    print("                    RESULTADOS OBTIDOS")
    print("=" * 75)
    print(f" • Status do Solver:      {status} (Ótimo Global Comprovado: {is_optimal})")
    print(f" • Tempo de Execução:     {result['runtime_sec']:.3f} segundos")
    print(f" • Custo da Ordem Inicial: {orig_cost:.4f}")
    print(f" • Custo da Ordem ÓTIMA:   {opt_cost:.4f}")
    
    improvement = ((orig_cost - opt_cost) / orig_cost) * 100.0
    print(f" • Redução de Atrito:     {improvement:.2f}% de melhoria nas transições!")
    print("-" * 75)
    
    print("\n[Comparativo Físico das Transições]")
    print(f"{'Métrica Avaliada':<35} | {'Original (Spotify)':<20} | {'Otimizado (HiGHS)':<20}")
    print("-" * 80)
    print(f"{'Salto Médio de BPM':<35} | {orig_breakdown['avg_bpm_jump']:>16.1f} BPM | {opt_breakdown['avg_bpm_jump']:>16.1f} BPM")
    print(f"{'Salto Máximo de BPM':<35} | {orig_breakdown['max_bpm_jump']:>16.1f} BPM | {opt_breakdown['max_bpm_jump']:>16.1f} BPM")
    print(f"{'Soma dos Saltos de BPM':<35} | {orig_breakdown['total_bpm_jump']:>16.1f} BPM | {opt_breakdown['total_bpm_jump']:>16.1f} BPM")
    print(f"{'Distância Média Camelot':<35} | {orig_breakdown['avg_camelot_dist']:>18.2f} | {opt_breakdown['avg_camelot_dist']:>18.2f}")
    print(f"{'Transições Harmônicas (dist <= 1)':<35} | {orig_breakdown['num_harmonic_transitions']:>15} de 35 | {opt_breakdown['num_harmonic_transitions']:>15} de 35")
    print(f"{'Trocas de Tom (Key Changes)':<35} | {orig_breakdown['n_key_changes']:>15} de 35 | {opt_breakdown['n_key_changes']:>15} de 35")
    print(f"{'Total de Quedas Bruscas de Energia':<35} | {orig_breakdown['total_energy_drops']:>18} | {opt_breakdown['total_energy_drops']:>18}")
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


if __name__ == '__main__':
    main()

