"""
Script de Execução Completa

Responsabilidades:
Orquestrar a execução completa da bateria experimental de Pesquisa Operacional:
  1. Executar 'generate_tables.py' para resolver os modelos no HiGHS e gerar todos os CSVs.
  2. Executar 'generate_figures.py' para gerar todos os gráficos.

Uso:
  python src/run_experiments.py
"""

import time
import generate_tables
import generate_figures


def main():
    t0 = time.time()

    # Gera todas as tabelas e dados numéricos
    generate_tables.main()

    # Gera todas as figuras
    generate_figures.main()

    total = time.time() - t0
    print("\n" + "=" * 80)
    print(f"   [SUCESSO] Pipeline completo concluído em {total:.1f} segundos!")
    print("=" * 80)


if __name__ == '__main__':
    main()
