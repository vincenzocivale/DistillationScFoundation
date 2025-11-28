import scanpy as sc
import os 
from omegaconf import OmegaConf as om
import anndata as ad
import pandas as pd
import os
import torch
import gc

from src.tahoe_x1.scripts.inference.predict_embeddings import predict_embeddings

def generate_embeddings(
    h5ad_path: str,
    output_dir: str = "/home/oem/vcivale/scFoundation/dataset/data_yuto/tahoe_result2/",
    model_size: str = "1b",
    batch_size: int = 8,
    gene_id_key: str = "gene_id",
    seq_len_dataset: int = 2048,
    return_gene_embeddings: bool = False ) -> sc.AnnData:
    """
    Wrapper per predict_embeddings che pre-configura il modello Tahoe-x1.

    Esegue l'inferenza su un file .h5ad e salva il risultato.

    Args:
        h5ad_path: Percorso al file di input .h5ad.
        output_dir: Cartella dove salvare i risultati.
        model_size: Dimensione del modello da Hugging Face (es. "1b" o "300m").
        gene_id_key: Chiave in adata.var per gli ID dei geni.
        seq_len_dataset: Lunghezza della sequenza per il modello.
        return_gene_embeddings: Se True, calcola anche gli embedding dei geni.

    Returns:
        L'oggetto AnnData con gli embedding in .obsm.
    """
    
    # Determina il nome del modello e dell'repo in base alla dimensione
    model_name_map = {
        "1b": "Tx1-1B",
        "700m": "Tx1-700M" # (Esempio, adatta se necessario)
    }
    model_name = model_name_map.get(model_size, f"Tx1-{model_size}")
    
    print(f"--- Avvio inferenza per {h5ad_path} ---")
    print(f"Modello: {model_name}, Output in: {output_dir}")

    # Costruisci la configurazione usando gli argomenti della funzione
    cfg_dict = {
        "model_name": model_name,
        "paths": {
            "hf_repo_id": "tahoebio/Tahoe-x1",
            "hf_model_size": model_size,
            "adata_input": h5ad_path,
        },
        "data": {
            "gene_id_key": gene_id_key
        },
        "predict": {
            "seq_len_dataset": seq_len_dataset,
            "return_gene_embeddings": return_gene_embeddings,
            "use_chem_inf": False,
             "batch_size": batch_size,
            "num_workers": 8,
            "prefetch_factor": 8,
        },
        "plot": {
            "save_dir": "./figures"
        },

    }
    
    # Crea l'oggetto OmegaConf
    cfg = om.create(cfg_dict)

    # Esegui l'inferenza
    adata = predict_embeddings(cfg)

    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva il file
    filename = os.path.basename(h5ad_path) 
    output_path = os.path.join(output_dir, filename)
    adata.write_h5ad(output_path)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    print(f"--- Risultato salvato in: {output_path} ---")
    
    return adata


def load_gene_mapping(mapping_path: str) -> dict:
    """
    Carica e processa il file di mapping da gene_name a gene_id.

    Gestisce i gene_name duplicati e pulisce i gene_id.
    """
    gencode_df = pd.read_csv(mapping_path, sep="\t")
    
    # --- Gestione Duplicati (Miglioramento Robustezza) ---
    # Controlla se ci sono nomi di geni (gene_name) duplicati
    n_duplicates = gencode_df.duplicated(subset=['gene_name']).sum()
    if n_duplicates > 0:
        print(f"Trovati {n_duplicates} 'gene_name' duplicati. "
                        "Verrà mantenuto solo il primo (keep='first').")
        # Rimuovi i duplicati per evitare un mapping ambiguo
        gencode_df = gencode_df.drop_duplicates(subset=['gene_name'], keep='first')

    # Rimuovi la versione dai gene_id (più efficiente con .str)
    gencode_df['gene_id'] = gencode_df['gene_id'].str.split('.').str[0]

    # Crea dizionario mapping (già efficiente nel tuo script)
    gene_name_to_id = dict(zip(gencode_df['gene_name'], gencode_df['gene_id']))
    
    print(f"Mapping caricato. {len(gene_name_to_id)} voci uniche.")
    return gene_name_to_id

def process_adata_file(filepath: str, output_path: str, gene_map: dict):
    """
    Applica il mapping dei geni a un singolo file .h5ad.
    """
    try:
        adata = ad.read_h5ad(filepath)
    except Exception as e:
        raise ValueError(f"Errore durante la lettura di {filepath}: {e}")

    # Mappa var_names (gene names) a Ensembl gene_id senza versione
    mapped_gene_ids = [gene_map.get(g, None) for g in adata.var_names]

    # Calcola statistiche di mapping (feedback utente)
    n_total = len(adata.var_names)
    mapped_mask = [x is not None for x in mapped_gene_ids]
    n_mapped = sum(mapped_mask)
    
    if n_total == 0:
        print(f"File {os.path.basename(filepath)} non ha geni. Saltato.")
        return
        
    print(f"[{os.path.basename(filepath)}] Mappati: {n_mapped}/{n_total} geni "
                 f"({n_mapped / n_total:.1%})")

    if n_mapped == 0:
        print(f"Nessun gene mappato per {os.path.basename(filepath)}. File non salvato.")
        return

    # Filtra solo quelli mappati (il .copy() è importante, corretto)
    adata = adata[:, mapped_mask].copy()

    # Salva gli Ensembl gene_id
    adata.var['gene_id'] = [x for x in mapped_gene_ids if x is not None]

    # Rimuovi la colonna '_index' se esiste
    if '_index' in adata.var.columns:
            adata.var = adata.var.drop(columns=['_index'])

    # Salva file aggiornato
    adata.write_h5ad(output_path)

