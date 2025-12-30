import scanpy as sc
import os
from typing import List, Optional


def load_h5ad_folder_lazy(folder_path: str, backed: bool = True, filename_filter: Optional[str] = None):
    """
    Carica tutti i file .h5ad in una cartella come un unico AnnData concatenato, usando la modalità backed per non saturare la RAM.
    Args:
        folder_path (str): Percorso della cartella contenente i file .h5ad
        backed (bool): Se True, usa la modalità backed (lettura su disco)
        filename_filter (Optional[str]): Se specificato, carica solo i file che contengono questa stringa nel nome
    Returns:
        AnnData: Oggetto AnnData concatenato
    """
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.h5ad') and (filename_filter is None or filename_filter in f)
    ]
    files.sort()
    if not files:
        raise FileNotFoundError(f"Nessun file .h5ad trovato in {folder_path}")

    # Carica in modalità backed e concatena
    adatas = [sc.read_h5ad(f, backed='r' if backed else None) for f in files]
    adata_concat = sc.concat(adatas, join='outer', axis=0, index_unique=None)
    return adata_concat
