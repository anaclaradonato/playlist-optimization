"""
Módulo de Carga e Limpeza de Dados (data_loader.py)
Projeto: Sequenciamento Ótimo de Playlists (Open ATSP) - Trabalho Final de PL

Responsabilidades:
1. Carregar o dataset 'spotify_songs.csv' (Kaggle).
2. Limpeza: descartar duplicatas dentro da mesma playlist, tempo=0 e key=-1.
3. Conversão das variáveis 'key' e 'mode' para a notação da Roda de Camelot (1A a 12B).
4. Fornecer funções para selecionar playlists reais por ID ou por faixa de tamanho (n).
"""

import os
import pandas as pd
from typing import Optional, Tuple, List


def transform_camelot(key: int, mode: int) -> Optional[str]:
    """
    Converte a tonalidade musical do Spotify (key: 0 a 11, mode: 0 ou 1)
    para o código correspondente na Roda de Camelot (1A a 12B).
    
    key: Classe de altura (0 = C, 1 = C#/Db, 2 = D, ..., 11 = B)
    mode: Modalidade (1 = Maior / B, 0 = Menor / A)
    """
    if pd.isna(key) or pd.isna(mode) or key < 0 or key > 11:
        return None
    
    key = int(key)
    mode = int(mode)
    
    # Mapeamento oficial da Roda de Camelot:
    # Tons Maiores (mode == 1 -> anel B)
    major_map = {
        0: '8B',   # C major
        1: '3B',   # C# / Db major
        2: '10B',  # D major
        3: '5B',   # D# / Eb major
        4: '12B',  # E major
        5: '7B',   # F major
        6: '2B',   # F# / Gb major
        7: '9B',   # G major
        8: '4B',   # G# / Ab major
        9: '11B',  # A major
        10: '6B',  # A# / Bb major
        11: '1B'   # B major
    }
    
    # Tons Menores (mode == 0 -> anel A)
    minor_map = {
        0: '5A',   # C minor
        1: '12A',  # C# / Db minor
        2: '7A',   # D minor
        3: '2A',   # D# / Eb minor
        4: '9A',   # E minor
        5: '4A',   # F minor
        6: '11A',  # F# / Gb minor
        7: '6A',   # G minor
        8: '1A',   # G# / Ab minor
        9: '8A',   # A minor
        10: '3A',  # A# / Bb minor
        11: '10A'  # B minor
    }
    
    if mode == 1:
        return major_map.get(key)
    elif mode == 0:
        return minor_map.get(key)
    return None


def load_clean_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Carrega o arquivo CSV 'spotify_songs.csv', aplica as regras de limpeza
    e adiciona a coluna 'camelot'.
    """
    if csv_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "..", "data", "spotify_songs.csv"),
            os.path.join(base_dir, "data", "spotify_songs.csv"),
            os.path.join("data", "spotify_songs.csv"),
            os.path.join(base_dir, "spotify_songs.csv"),
            "spotify_songs.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_path = os.path.abspath(c)
                break
            
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo CSV não encontrado em: {csv_path}")
        
    print(f"-> Carregando dataset: {csv_path}...")
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    
    # 1. Filtragem de inconsistências acústicas (tempo > 0 e key >= 0)
    df = df[(df['tempo'] > 0) & (df['key'] >= 0) & (df['energy'] >= 0)].copy()
    
    # 2. Descarte de duplicatas dentro da mesma playlist
    df = df.drop_duplicates(subset=['playlist_id', 'track_id']).copy()

    df = df.drop_duplicates(subset=['playlist_id', 'track_name', 'track_artist']).copy()
    
    # 3. Conversão para a Roda de Camelot
    df['camelot'] = df.apply(lambda row: transform_camelot(row['key'], row['mode']), axis=1)
    
    # Remove eventuais linhas com tom nulo
    df = df.dropna(subset=['camelot']).copy()
    
    final_count = len(df)
    num_playlists = df['playlist_id'].nunique()
    print(f"-> Limpeza concluída: {final_count} faixas válidas em {num_playlists} playlists reais (descartadas {initial_count - final_count} linhas inconsistentes/duplicadas).")
    
    return df


def get_playlist_by_id(df: pd.DataFrame, playlist_id: str) -> pd.DataFrame:
    """
    Retorna o subset de músicas de uma playlist específica, resetando o índice (0 a n-1).
    """
    pl_df = df[df['playlist_id'] == playlist_id].copy().reset_index(drop=True)
    if len(pl_df) == 0:
        raise ValueError(f"Playlist com ID '{playlist_id}' não encontrada no dataset.")
    return pl_df


def list_sample_playlists(df: pd.DataFrame, target_sizes: List[int] = [36, 40, 60, 75, 100]) -> pd.DataFrame:
    """
    Retorna um resumo de playlists candidatas para os testes com tamanhos próximos aos especificados.
    """
    pl_summary = (
        df.groupby(['playlist_id', 'playlist_name', 'playlist_genre'])
        .size()
        .reset_index(name='num_tracks')
    )
    
    samples = []
    for target in target_sizes:
        # Busca exata ou a mais próxima
        exact_matches = pl_summary[pl_summary['num_tracks'] == target]
        if len(exact_matches) > 0:
            samples.append(exact_matches.iloc[0])
        else:
            diff = (pl_summary['num_tracks'] - target).abs()
            closest = pl_summary.loc[diff.idxmin()]
            samples.append(closest)
            
    return pd.DataFrame(samples).drop_duplicates(subset=['playlist_id']).reset_index(drop=True)


if __name__ == '__main__':
    # Teste rápido do carregador
    df = load_clean_dataset()
    samples = list_sample_playlists(df, target_sizes=[36, 40, 60, 75, 100])
    print("\nPlaylists sugeridas para os experimentos:")
    print(samples.to_string(index=False))

