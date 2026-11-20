"""
Módulo de Geração de Figuras e Gráficos Científicos

Responsabilidades:
Ler as tabelas de dados na pasta 'results/' e renderizar todos os 4 gráficos visuais:
  1. grafico_figura_de_escala.png     
  2. grafico_estudo_de_caso_demo.png  
  3. grafico_pareto_multiobjetivo.png 
  4. grafico_dispersao_acustica.png   
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import load_clean_dataset, get_playlist_by_id
from cost_matrix import calc_camelot_cost


def get_results_dir():
    """Retorna o caminho absoluto do diretório de resultados."""
    base = os.path.dirname(os.path.abspath(__file__))
    target = os.path.abspath(os.path.join(base, "..", "results"))
    os.makedirs(target, exist_ok=True)
    return target


def generate_scale_figure(res_dir, limite_tempo=300.0):
    """
    Figura 1 do Artigo: Três painéis
    1. Tempo de CPU x Tamanho (HiGHS vs Heurística em log10)
    2. Heurística: diferença para o HiGHS (%)
    3. Redução do custo x Ordem aleatória (%)
    """
    csv_path = os.path.join(res_dir, 'tabela_resultados_experimentos.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Tabela não encontrada: {csv_path}. Execute 'generate_tables.py' primeiro.")

    df_bench = pd.read_csv(csv_path)
    n = df_bench['Tamanho_n'].astype(int).values
    t_exato = df_bench['Tempo_HiGHS_s'].astype(float).values
    t_heur = df_bench['Tempo_Heur_s'].astype(float).values

    def texto_para_numero(valor):
        try:
            return float(str(valor).replace('%', '').replace('+', ''))
        except ValueError:
            return float('nan')

    dif = np.array([texto_para_numero(v) for v in df_bench['GAP_Heur_%']])
    reducao = df_bench['Reducao_vs_Shuffle_%'].astype(float).values
    generos = df_bench['Genero'].values

    provado = [('Ótimo' in str(s)) or (str(s) == 'Optimal') for s in df_bench['Status_HiGHS']]
    posicoes = np.arange(len(n))
    rotulos = [f"n={a}\n({g})" for a, g in zip(n, generos)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Painel 1: Tempo de CPU x Tamanho
    ax1.plot(n, t_exato, color='#0284c7', linewidth=2.0, label='HiGHS (modelo exato, MTZ)')
    for x, y, ok in zip(n, t_exato, provado):
        ax1.plot(x, y, marker='s', markersize=7, color='#0284c7',
                 markerfacecolor='#0284c7' if ok else 'white')
    ax1.plot(n, t_heur, color='#10b981', linewidth=2.0, linestyle='--', marker='o',
             markersize=6, label='Heurística (vizinho mais próximo + Or-opt)')
    ax1.axhline(limite_tempo, color='#e11d48', linestyle=':', linewidth=1.5,
                label=f'Tempo limite ({limite_tempo:.0f} s)')
    ax1.annotate(f"{t_exato[-1]:.0f} s", (n[-1], t_exato[-1]), textcoords='offset points',
                 xytext=(-6, -15), ha='right', fontsize=8.5, color='#075985')
    ax1.annotate(f"{t_heur[-1]:.1f} s", (n[-1], t_heur[-1]), textcoords='offset points',
                 xytext=(-4, 8), ha='right', fontsize=8.5, color='#047857')
    ax1.set_yscale('log')
    ax1.set_ylim(top=max(limite_tempo, t_exato.max()) * 4)
    ax1.set_xticks(n)
    ax1.set_xticklabels(n, fontsize=8)
    ax1.set_title('Tempo de CPU x Tamanho', fontsize=10.5, fontweight='bold')
    ax1.set_xlabel('Número de faixas (n)', fontsize=9)
    ax1.set_ylabel('Tempo (segundos, escala log)', fontsize=9)
    ax1.legend(fontsize=7.8, loc='lower right', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.6, which='both')

    # Painel 2: Heurística: diferença para o HiGHS
    ax2.bar(posicoes, dif, width=0.6, color='#8b5cf6', alpha=0.9, edgecolor='#6d28d9')
    for p, v in zip(posicoes, dif):
        if not np.isnan(v):
            ax2.annotate(f"{v:+.1f}%", (p, v), textcoords='offset points', xytext=(0, 3),
                         ha='center', fontsize=8.5, fontweight='bold', color='#5b21b6')
    media = np.nanmean(dif)
    ax2.axhline(media, color='#ec4899', linestyle='--', linewidth=1.6,
                label=f'Média das {len(n)} instâncias = {media:.1f}%')
    ax2.set_xticks(posicoes)
    ax2.set_xticklabels(rotulos, fontsize=7.8)
    ax2.set_ylim(0, np.nanmax(dif) * 1.25)
    ax2.set_title('Heurística: diferença para o HiGHS', fontsize=10.5, fontweight='bold')
    ax2.set_ylabel('Custo da heurística acima do HiGHS (%)', fontsize=9)
    ax2.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.6, axis='y')

    # Painel 3: Redução do custo x Ordem aleatória
    ax3.bar(posicoes, reducao, width=0.6, color='#0ea5e9', alpha=0.9, edgecolor='#0369a1')
    for p, v in zip(posicoes, reducao):
        ax3.annotate(f"{v:.0f}%", (p, v), textcoords='offset points', xytext=(0, 3),
                     ha='center', fontsize=8.5, fontweight='bold', color='#075985')
    ax3.set_xticks(posicoes)
    ax3.set_xticklabels(rotulos, fontsize=7.8)
    ax3.set_ylim(0, 100)
    ax3.set_title('Redução do custo x Ordem aleatória', fontsize=10.5, fontweight='bold')
    ax3.set_ylabel('Redução do custo total (%)', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.6, axis='y')

    n_prov = sum(provado)
    fig.text(0.5, 0.005,
             f"Uma playlist por tamanho (gêneros diferentes, por isso o tempo não cresce de forma regular). "
             f"Ótimo provado em {n_prov} de {len(n)} instâncias (marcador cheio; tolerância padrão do HiGHS de 0,01%). "
             f"Redução medida contra a média de 500 ordens aleatórias.",
             ha='center', fontsize=7.8, style='italic', color='#64748b')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = os.path.join(res_dir, 'grafico_figura_de_escala.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"   [OK] Salvo: {out_path}")
    return out_path


def generate_pareto_frontier_plot(res_dir):
    """Figura 3 do Artigo: Curva de Pareto Ritmo vs Harmonia com destaques limpos."""
    csv_path = os.path.join(res_dir, 'tabela_pareto_sensibilidade.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Tabela não encontrada: {csv_path}. Execute 'generate_tables.py' primeiro.")

    df_pts = pd.read_csv(csv_path)
    df_nondom = df_pts[df_pts['is_nondominated']].sort_values('avg_bpm_jump')
    df_dom = df_pts[~df_pts['is_nondominated']]

    plt.figure(figsize=(9, 5.2), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Curva não-dominada
    plt.plot(df_nondom['avg_bpm_jump'], df_nondom['avg_camelot_dist'], color='#0284c7', linestyle='-',
             linewidth=2.0, marker='o', markersize=6, label='Fronteira de Pareto')

    # 2. Destaque do Caso-Base (0.35 / 0.35 / 0.30)
    base_match = df_nondom[np.isclose(df_nondom['w_bpm'], 0.35)]
    if len(base_match) > 0:
        base_pt = base_match.iloc[0]
        plt.scatter([base_pt['avg_bpm_jump']], [base_pt['avg_camelot_dist']], color='#7c3aed', s=140,
                    zorder=6, marker='*', label='Caso-Base Adotado (w_bpm=0.35, w_tom=0.35, w_ene=0.30)')

    # 3. Anotações dos pontos
    for _, row in df_nondom.iterrows():
        wb_val = row['w_bpm']
        wt_val = row['w_tom']
        plt.annotate(
            f"{wb_val:.2f} / {wt_val:.2f}",
            (row['avg_bpm_jump'], row['avg_camelot_dist']),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7.5,
            color='#1e293b'
        )

    plt.title("Fronteira de Pareto: Trade-off Ritmo vs. Harmonia (com w_energia = 0.30 fixo)", fontsize=10.5, fontweight='bold', pad=12, color='#0f172a')
    plt.xlabel("Atrito Rítmico Médio (Salto em BPM)", fontsize=9, fontweight='bold', color='#334155')
    plt.ylabel("Atrito harmônico Médio (Distância Camelot/7)", fontsize=9, fontweight='bold', color='#334155')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5, loc='upper right')

    plt.figtext(0.5, 0.01, "* Nota: Projeção bidimensional com w_energia = 0.30 constante (caso-base). A curva demonstra a flexibilidade do modelo multiobjetivo.",
                ha='center', fontsize=7.8, style='italic', color='#64748b')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = os.path.join(res_dir, 'grafico_pareto_multiobjetivo.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"   [OK] Salvo: {out_path}")
    return out_path


def generate_case_study_demo_plot(df_all, res_dir, playlist_id="0DdGqNa18DwYyfIR05OrW1"):
    """Figura 2 do Artigo: Estudo de Caso faixa a faixa com 3 painéis verticais (BPM, Tom Camelot e Energia)."""
    seq_path = os.path.join(res_dir, 'sequencia_otimizada.csv')
    df_pl = get_playlist_by_id(df_all, playlist_id)
    n_tr = len(df_pl)

    tempos = df_pl['tempo'].values
    camelots = df_pl['camelot'].values
    energies = df_pl['energy'].values

    # Ordem original do arquivo CSV: faixas 0 a n_tr-1
    orig_route = list(range(n_tr))

    # Ordem otimizada lida de sequencia_otimizada.csv
    if os.path.exists(seq_path):
        df_seq = pd.read_csv(seq_path)
        name_to_idx = {name: i for i, name in enumerate(df_pl['track_name'].values)}
        opt_route = [name_to_idx[name] for name in df_seq['Musica'].values if name in name_to_idx]
        if len(opt_route) != n_tr:
            opt_route = list(range(n_tr))
    else:
        opt_route = list(range(n_tr))

    # Extrai métricas transição a transição
    def extract_metrics(route):
        b_list, t_list, e_list = [], [], []
        for k in range(n_tr - 1):
            i, j = route[k], route[k + 1]
            dt = min(abs(tempos[i] - tempos[j]), abs(tempos[i] - 2.0 * tempos[j]), abs(2.0 * tempos[i] - tempos[j]))
            cd = int(round(calc_camelot_cost(camelots[i], camelots[j]) * 7.0))
            de = abs(energies[j] - energies[i])
            b_list.append(dt)
            t_list.append(cd)
            e_list.append(de)
        return b_list, t_list, e_list

    bpm_orig, tom_orig, ene_orig = extract_metrics(orig_route)
    bpm_opt, tom_opt, ene_opt = extract_metrics(opt_route)

    # Valores esperados analíticos sobre todos os n*(n-1) pares ordenados da playlist
    all_bpm, all_tom, all_ene = [], [], []
    for i in range(n_tr):
        for j in range(n_tr):
            if i != j:
                dt = min(abs(tempos[i] - tempos[j]), abs(tempos[i] - 2.0 * tempos[j]), abs(2.0 * tempos[i] - tempos[j]))
                cd = int(round(calc_camelot_cost(camelots[i], camelots[j]) * 7.0))
                all_bpm.append(dt)
                all_tom.append(cd)
                all_ene.append(abs(energies[j] - energies[i]))
    exp_bpm = float(np.mean(all_bpm))
    exp_tom = float(np.mean(all_tom))
    exp_ene = float(np.mean(all_ene))

    harm_orig = sum(1 for d in tom_orig if d <= 1)
    harm_opt = sum(1 for d in tom_opt if d <= 1)

    # 3 painéis empilhados verticalmente
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.5, 10.2), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    transitions = list(range(1, n_tr))

    # Painel 1: Salto Rítmico de BPM (Headroom até 88 BPM para a legenda não cobrir a curva)
    ax1.plot(transitions, bpm_orig, color='#94a3b8', linestyle='--', marker='o', markersize=4.2, linewidth=1.4,
             label=f'Ordem do Arquivo (Média: {np.mean(bpm_orig):.1f} BPM)')
    ax1.plot(transitions, bpm_opt, color='#0284c7', linestyle='-', marker='s', markersize=4.5, linewidth=2.0,
             label=f'Otimizado HiGHS (Média: {np.mean(bpm_opt):.1f} BPM)')
    ax1.axhline(exp_bpm, color='#e11d48', linestyle=':', linewidth=1.5,
                label=f'Valor Esperado Aleatório: {exp_bpm:.1f} BPM')
    ax1.set_title('Salto Rítmico Efetivo de BPM: Δ_BPM = min(|a-b|, |a-2b|, |2a-b|)', fontsize=9.8, fontweight='bold', pad=8, color='#0f172a')
    ax1.set_ylabel('Salto Efetivo (BPM)', fontsize=8.8, fontweight='bold', color='#334155')
    ax1.set_xlim(0.5, n_tr - 0.5)
    ax1.set_ylim(0, 88)
    ax1.set_xticks(range(1, n_tr, 2))
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.92, edgecolor='#cbd5e1', fontsize=8.0, loc='upper right')

    # Painel 2: Distância Harmônica na Roda de Camelot (Headroom até 10.5 para a legenda ficar acima de y=7)
    ax2.plot(transitions, tom_orig, color='#94a3b8', linestyle='--', marker='o', markersize=4.2, linewidth=1.4,
             label=f'Ordem do Arquivo (Média: {np.mean(tom_orig):.2f} passos | {harm_orig}/35 harmônicas)')
    ax2.plot(transitions, tom_opt, color='#8b5cf6', linestyle='-', marker='s', markersize=4.5, linewidth=2.0,
             label=f'Otimizado HiGHS (Média: {np.mean(tom_opt):.2f} passos | {harm_opt}/35 harmônicas)')
    ax2.axhline(exp_tom, color='#e11d48', linestyle=':', linewidth=1.5,
                label=f'Valor Esperado Aleatório: {exp_tom:.2f} passos')
    ax2.set_title('Distância Harmônica na Roda de Camelot (0 a 7 passos)', fontsize=9.8, fontweight='bold', pad=8, color='#0f172a')
    ax2.set_ylabel('Passos Camelot', fontsize=8.8, fontweight='bold', color='#334155')
    ax2.set_xlim(0.5, n_tr - 0.5)
    ax2.set_ylim(-0.3, 10.5)
    ax2.set_yticks(range(0, 8))
    ax2.set_xticks(range(1, n_tr, 2))
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.92, edgecolor='#cbd5e1', fontsize=8.0, loc='upper right')

    # Painel 3: Salto de Intensidade de Energia (|ΔE|) (Headroom até 0.54 para a legenda ficar acima do pico de 0.34)
    ax3.plot(transitions, ene_orig, color='#94a3b8', linestyle='--', marker='o', markersize=4.2, linewidth=1.4,
             label=f'Ordem do Arquivo (Média: {np.mean(ene_orig):.2f} | Máx: {np.max(ene_orig):.2f})')
    ax3.plot(transitions, ene_opt, color='#10b981', linestyle='-', marker='s', markersize=4.5, linewidth=2.0,
             label=f'Otimizado HiGHS (Média: {np.mean(ene_opt):.2f} | Máx: {np.max(ene_opt):.2f})')
    ax3.axhline(exp_ene, color='#e11d48', linestyle=':', linewidth=1.5,
                label=f'Valor Esperado Aleatório: {exp_ene:.2f}')
    ax3.set_title('Magnitude do Salto de Intensidade de Energia (|ΔE|)', fontsize=9.8, fontweight='bold', pad=8, color='#0f172a')
    ax3.set_xlabel('Transição Consecutiva entre Faixas (k → k+1)', fontsize=9.0, fontweight='bold', color='#334155')
    ax3.set_ylabel('Magnitude |ΔE|', fontsize=8.8, fontweight='bold', color='#334155')
    ax3.set_xlim(0.5, n_tr - 0.5)
    ax3.set_ylim(0, 0.54)
    ax3.set_xticks(range(1, n_tr, 2))
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(frameon=True, facecolor='white', framealpha=0.92, edgecolor='#cbd5e1', fontsize=8.0, loc='upper right')

    fig.suptitle('Estudo de Caso: Transições Faixa a Faixa — Playlist "Viral Southern Hip Hop" (n = 36)', fontsize=11.2, fontweight='bold', y=0.993, color='#0f172a')
    fig.text(0.5, 0.012, '* Nota: Uma playlist, uma execução; pesos BPM/tom/energia = 0,35/0,35/0,30 e γ = 1,5. Pares como 98 ↔ 196 BPM contam como transição suave (dobro/metade). Linha pontilhada vermelha representa o valor esperado analítico sob permutação uniforme.',
             ha='center', fontsize=7.4, style='italic', color='#64748b')

    plt.tight_layout(rect=[0, 0.025, 1, 0.985])
    out_path = os.path.join(res_dir, 'grafico_estudo_de_caso_demo.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"   [OK] Salvo: {out_path}")
    return out_path


def generate_acoustic_dispersion_plot(df_all, res_dir, playlist_id="0DdGqNa18DwYyfIR05OrW1"):
    """Gráfico de Reserva: Heterogeneidade acústica intra-gênero (BPM vs Energia e Roda de Camelot)."""
    df_pl = get_playlist_by_id(df_all, playlist_id)
    n = len(df_pl)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. BPM vs Energia
    colors = ['#0284c7' if str(c).endswith('B') else '#e11d48' for c in df_pl['camelot']]
    ax1.scatter(df_pl['tempo'], df_pl['energy'], c=colors, s=65, alpha=0.85, edgecolors='#0f172a', linewidth=0.8)
    ax1.set_title(f'Espaço Acústico: Andamento (BPM) vs. Energia\n({n} faixas — "Viral Southern Hip Hop")', fontsize=9.5, fontweight='bold', pad=10, color='#0f172a')
    ax1.set_xlabel(f"Andamento (BPM) [Min: {df_pl['tempo'].min():.1f} | Max: {df_pl['tempo'].max():.1f}]", fontsize=8.5, fontweight='bold', color='#334155')
    ax1.set_ylabel(f"Intensidade Acústica (Energia) [Min: {df_pl['energy'].min():.2f} | Max: {df_pl['energy'].max():.2f}]", fontsize=8.5, fontweight='bold', color='#334155')
    ax1.axvline(df_pl['tempo'].mean(), color='#64748b', linestyle=':', linewidth=1.2, label=f"Média BPM = {df_pl['tempo'].mean():.1f}")
    ax1.axhline(df_pl['energy'].mean(), color='#64748b', linestyle='--', linewidth=1.2, label=f"Média Energia = {df_pl['energy'].mean():.2f}")
    ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Distribuição Camelot
    key_counts = df_pl['camelot'].value_counts().sort_index()
    bar_colors = ['#0284c7' if c.endswith('B') else '#e11d48' for c in key_counts.index]
    ax2.bar(range(len(key_counts)), key_counts.values, color=bar_colors, edgecolor='#0f172a', linewidth=0.6, alpha=0.85)
    ax2.set_title(f'Diversidade Harmônica na Roda de Camelot\n({len(key_counts)} códigos distintos presentes em {n} faixas)', fontsize=9.5, fontweight='bold', pad=10, color='#0f172a')
    ax2.set_xlabel('Código Camelot (Azul: Maior / B, Vermelho: Menor / A)', fontsize=8.5, fontweight='bold', color='#334155')
    ax2.set_ylabel('Quantidade de Músicas', fontsize=8.5, fontweight='bold', color='#334155')
    ax2.set_xticks(range(len(key_counts)))
    ax2.set_xticklabels(key_counts.index, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(0, max(key_counts.values) + 2))
    ax2.grid(True, linestyle=':', alpha=0.6, axis='y')

    plt.tight_layout()
    out_path = os.path.join(res_dir, 'grafico_dispersao_acustica.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"   [OK] Salvo: {out_path}")
    return out_path


def main():
    res_dir = get_results_dir()
    df_all = load_clean_dataset()

    generate_scale_figure(res_dir, limite_tempo=300)
    generate_pareto_frontier_plot(res_dir)
    generate_case_study_demo_plot(df_all, res_dir)
    generate_acoustic_dispersion_plot(df_all, res_dir)

    print("\n" + "=" * 80)
    print("   [CONCLUÍDO] Todos os 4 gráficos PNG foram gerados com sucesso!")
    print(f"   Diretório: {res_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

