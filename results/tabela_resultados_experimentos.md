# Resultados da Bateria Experimental (HiGHS vs. Heurística vs. Linha de Base)

|   Tamanho_n | Genero   | Nome_Playlist           |   Custo_Ordem_CSV |   Shuffle_Medio |   Shuffle_DesvPad |   Tempo_Heur_s |   Tempo_HiGHS_s |   Speedup_Heur_x |   Custo_Heur |   Custo_HiGHS |   Dual_Bound_HiGHS | Status_HiGHS   | Gap_HiGHS_%   | GAP_Heur_%   |   Reducao_vs_Shuffle_% |   Reducao_vs_CSV_% |
|------------:|:---------|:------------------------|------------------:|----------------:|------------------:|---------------:|----------------:|-----------------:|-------------:|--------------:|-------------------:|:---------------|:--------------|:-------------|-----------------------:|-------------------:|
|          15 | r&b      | Fresh Essentials        |            6.2011 |          6.1844 |            0.5245 |         0.0063 |           0.45  |             71.7 |       3.1911 |        3.1042 |             3.1042 | Optimal        | 0.00%         | +2.80%       |                  49.81 |              49.94 |
|          30 | rap      | Gangsta Rap 💎 Rap Part |           10.741  |         11.9598 |            0.7481 |         0.0367 |           1.206 |             32.8 |       4.6927 |        4.6106 |             4.6106 | Optimal        | 0.00%         | +1.78%       |                  61.45 |              57.08 |
|          36 | rap      | Viral Southern Hip Hop  |           15.7015 |         14.7792 |            0.8658 |         0.0843 |           1.792 |             21.3 |       5.8848 |        5.5383 |             5.5383 | Optimal        | 0.00%         | +6.26%       |                  62.53 |              64.73 |
|          50 | edm      | SINULOG Festival 2020   |           18.9545 |         20.9779 |            1.0442 |         0.2239 |          24.964 |            111.5 |       7.3915 |        6.9192 |             6.9192 | Optimal        | 0.00%         | +6.82%       |                  67.02 |              63.5  |
|          60 | pop      | Post Teen Pop           |           23.4075 |         25.0059 |            1.1378 |         0.7514 |          23.729 |             31.6 |       8.8882 |        8.6847 |             8.6847 | Optimal        | 0.00%         | +2.34%       |                  65.27 |              62.9  |
|          75 | pop      | Christian Dance Party   |           30.274  |         29.5692 |            1.1498 |         1.331  |          40.683 |             30.6 |       8.9033 |        8.383  |             8.383  | Optimal        | 0.00%         | +6.21%       |                  71.65 |              72.31 |
|         100 | r&b      | Groovy // Funky // Neo  |           32.3199 |         34.9782 |            1.1761 |         5.0791 |         191.866 |             37.8 |      10.1921 |        9.8119 |             9.8119 | Optimal        | 0.00%         | +3.87%       |                  71.95 |              69.64 |

---
### Notas Metodológicas e Esclarecimentos Acadêmicos:
1. **Status do Solver e Otimalidade Global:**
   - `Optimal`: O solver HiGHS comprovou matematicamente o ótimo global da formulação Open ATSP (gap do solver = 0,00%) em **todas as 7 instâncias**, desde a menor (15 faixas) até a maior existente no dataset (100 faixas).
2. **Convergência Temporal da Instância n = 100 faixas:**
   - Na maior playlist do dataset ($n = 100$), o HiGHS encontrou uma solução viável de 10,2864 aos 60 s (gap de 5,98%) e provou a otimalidade global em **199,4 s (~3 min e 19 s)** com custo mínimo de **9,8119** (gap de 0,00%).
   - A heurística rápida (Nearest Neighbor Multi-Start + Or-opt) encontrou uma solução quase-ótima de 10,1921 em apenas **5,0 s** (GAP de apenas +3,88% em relação ao ótimo global provado, com speedup de ~40x frente ao solver exato).
3. **Identificação das Linhas de Base:**
   - `Custo_Ordem_CSV`: Refere-se à ordem das faixas registrada no arquivo CSV do dataset Kaggle.
   - `Shuffle_Medio` e `Shuffle_DesvPad`: Média e desvio padrão de 500 permutações aleatórias (`seed=0`), estabelecendo o controle estatístico neutro na ausência de curadoria acústica.
4. **Interpretação do Custo Acústico:**
   - A redução de custo quantifica a otimização da função objetivo matemática multiobjetivo ($c_{ij}$), servindo como proxy computacional de continuidade de andamento, harmonia e intensidade.
