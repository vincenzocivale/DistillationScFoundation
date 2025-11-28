

import os
import gc
import uuid
import shutil
import numpy as np
import pandas as pd
import scanpy as sc

from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from datasets.arrow_writer import ArrowWriter

import polars as pl
import scanpy as sc
import numpy as np
import os

import os
import numpy as np
import polars as pl
import scanpy as sc

def create_embedding_polars_dataset(
    folder_path: str,
    key_embedding: str,
    labels_csv_path: str,
    cell_id_column_csv: str = "Cell_id",
    label_column_csv: str = "Detailed_Cluster_names",
    cell_id_in_h5ad: str = "cell_id",
):

    # 1. Leggi CSV label con Polars
    labels_df = pl.read_csv(labels_csv_path).with_columns([
        pl.col(cell_id_column_csv).cast(pl.Utf8)
    ])

    # Set di ID cellule nel CSV
    csv_cell_ids = set(labels_df.select(cell_id_column_csv).to_series().to_list())

    # 2. Lista per contenere i df di embeddings + cell_id per ogni h5ad
    embedding_dfs = []

    h5_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".h5ad"))

    all_h5ad_cell_ids = set()

    for fname in h5_files:
        path = os.path.join(folder_path, fname)
        base_name = os.path.splitext(fname)[0]

        adata = sc.read_h5ad(path, backed='r')

        if key_embedding not in adata.obsm.keys():
            raise ValueError(f"Embedding `{key_embedding}` non trovato in {fname}")

        emb = adata.obsm[key_embedding][...]  # backed load
        n = emb.shape[0]

        if cell_id_in_h5ad is None:
            cell_ids = adata.obs_names.astype(str).values
        else:
            if cell_id_in_h5ad not in adata.obs.columns:
                raise ValueError(f"Colonna `{cell_id_in_h5ad}` non trovata in adata.obs per {fname}")
            cell_ids = adata.obs[cell_id_in_h5ad].astype(str).values

        all_h5ad_cell_ids.update(cell_ids)

        # Crea DataFrame Polars temporaneo
        tmp_df = pl.DataFrame({
            "cell_id": cell_ids,
            "embedding": [emb[i].astype(np.float32) for i in range(n)],
            "source_file": [base_name] * n
        })

        embedding_dfs.append(tmp_df)

    # Controllo mismatch cellule
    missing_in_csv = all_h5ad_cell_ids - csv_cell_ids
    missing_in_h5ad = csv_cell_ids - all_h5ad_cell_ids

    if missing_in_csv:
        raise ValueError(f"Cellule presenti negli h5ad ma non nel CSV: {missing_in_csv}")

    if missing_in_h5ad:
        raise ValueError(f"Cellule presenti nel CSV ma non negli h5ad: {missing_in_h5ad}")

    # 3. Concateno tutti i dataframe embeddings
    all_embeddings_df = pl.concat(embedding_dfs)

    # 4. Fai join con labels
    merged_df = all_embeddings_df.join(
        labels_df.select([cell_id_column_csv, label_column_csv]),
        left_on="cell_id",
        right_on=cell_id_column_csv,
        how="left"
    )

    # 5. Gestisci label mancanti: sostituisci null con "Unknown" e rinomina la colonna in "label"
    merged_df = merged_df.with_columns(
        pl.col(label_column_csv).fill_null("Unknown").alias("label")
    ).drop(label_column_csv)

    return merged_df




def create_embedding_dataset_from_h5ad_lazy(
    folder_path: str,
    key_embedding: str,
    labels_csv_path: str,
    cell_id_column_csv: str = "Cell_id",
    label_column_csv: str = "Detailed_Cluster_names",
    cell_id_in_h5ad: str = "cell_id",
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    cache_path: str | None = None,
    arrow_temp_path: str = "temporary_arrow_dataset",
):
    """
    Versione riscritta in modo intelligente e scalabile:
    - Usa ArrowWriter (streaming) → NO reloading, NO concatenazioni ripetute.
    - RAM usage minimo.
    - Più veloce, più leggibile, atomic-safe.
    """

    # ================================================================
    # 0. CACHE CHECK
    # ================================================================
    if cache_path is not None and os.path.exists(cache_path):
        print(f"[CACHE] Carico dataset da cache: {cache_path}")
        return load_from_disk(cache_path)

    # ================================================================
    # 1. Carico CSV con le label
    # ================================================================
    labels_df = (
        pd.read_csv(labels_csv_path)
        .assign(**{cell_id_column_csv: lambda df: df[cell_id_column_csv].astype(str)})
        .set_index(cell_id_column_csv)[label_column_csv]
    )

    # ================================================================
    # 2. Preparo directory Arrow di output
    # ================================================================
    if os.path.exists(arrow_temp_path):
        shutil.rmtree(arrow_temp_path)
    os.makedirs(arrow_temp_path, exist_ok=True)

    arrow_path = os.path.join(arrow_temp_path, "data.arrow")

    writer = ArrowWriter(
        path=arrow_path,
        schema=None,  # Lo inferisce al primo batch
        writer_batch_size=10_000,
    )

    print("📦 ArrowWriter inizializzato (streaming mode).")

    # ================================================================
    # 3. Itero i file h5ad e scrivo direttamente su Arrow
    # ================================================================
    h5_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".h5ad"))
    print(f"Trovati {len(h5_files)} file h5ad.\n")

    for fname in h5_files:
        print(f"[READ] {fname}")
        path = os.path.join(folder_path, fname)
        base_name = os.path.splitext(fname)[0]

        # Lazy loading (backed)
        adata = sc.read_h5ad(path, backed="r")

        if key_embedding not in adata.obsm.keys():
            raise ValueError(f"Embedding `{key_embedding}` non trovato in {fname}")

        emb = adata.obsm[key_embedding][...]  # backed, lazy
        n = emb.shape[0]

        # Cell IDs
        if cell_id_in_h5ad is None:
            cell_ids = adata.obs_names.astype(str).values
        else:
            if cell_id_in_h5ad not in adata.obs.columns:
                raise ValueError(
                    f"Colonna `{cell_id_in_h5ad}` non presente in `adata.obs` per {fname}"
                )
            cell_ids = adata.obs[cell_id_in_h5ad].astype(str).values

        # Label (allineamento su CSV)
        labels = labels_df.reindex(cell_ids).fillna("Unknown").astype(str).values

        # Scrittura batch
        batch = {
            "embedding": [emb[i].astype(np.float32) for i in range(n)],
            "label": labels.tolist(),
            "cell_id": cell_ids.tolist(),
            "source_file": [base_name] * n,
        }
        writer.write(batch)

        del adata, emb, labels, cell_ids, batch
        gc.collect()

    # ================================================================
    # 4. Chiudo ArrowWriter e ricarico dataset
    # ================================================================
    writer.finalize()
    full_dataset = Dataset.from_file(arrow_path)

    print(f"\n📊 Totale celle nel dataset: {len(full_dataset)}")

    # ================================================================
    # 5. Split stratificato (globale)
    # ================================================================
    split_test = full_dataset.train_test_split(
        test_size=test_size,
        stratify_by_column="label",
        seed=random_state,
    )

    rel_val_size = val_size / (1 - test_size)

    split_val = split_test["train"].train_test_split(
        test_size=rel_val_size,
        stratify_by_column="label",
        seed=random_state,
    )

    dataset = DatasetDict(
        {
            "train": split_val["train"],
            "validation": split_val["test"],
            "test": split_test["test"],
        }
    )

    # ================================================================
    # 6. Salvataggio finale in cache
    # ================================================================
    if cache_path is not None:
        print(f"[CACHE] Salvo dataset finale in {cache_path}")
        dataset.save_to_disk(cache_path)

    print("🎉 Dataset creato con successo!")
    return dataset





def create_hf_dataset_from_h5ad_and_npy(
    h5ad_folder: str,
    npy_folder: str,
    labels_csv_path: str,
    label_column_csv: str = "Detailed_Cluster_names",
    cell_id_column_csv: str = "Cell_id",
    cell_id_in_h5ad: str = None,   # None per usare adata.obs_names
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """
    Crea dataset HF da file .h5ad in una cartella + file .npy embeddings in un'altra, 
    più CSV con label esterne.

    Args:
        h5ad_folder: percorso contenente file .h5ad
        npy_folder: percorso contenente file .npy corrispondenti (stesso base nome)
        labels_csv_path: CSV con colonne cell_id e label
        label_column_csv: nome colonna label nel CSV
        cell_id_column_csv: nome colonna ID cellula nel CSV
        cell_id_in_h5ad: nome colonna ID in adata.obs; se None usa adata.obs_names
        test_size: frazione test split
        val_size: frazione validazione sul train restante
        random_state: seed per split

    Returns:
        dict con chiavi 'train', 'validation', 'test' → HF Dataset
    """

    # Carico CSV label
    labels_df = pd.read_csv(labels_csv_path)
    labels_df[cell_id_column_csv] = labels_df[cell_id_column_csv].astype(str)
    labels_df = labels_df.set_index(cell_id_column_csv)

    all_datasets = []

    # Trovo file h5ad
    h5ad_files = sorted([f for f in os.listdir(h5ad_folder) if f.endswith(".h5ad")])

    for h5_file in h5ad_files:
        base_name = os.path.splitext(h5_file)[0]
        npy_file = base_name + ".npy"
        npy_path = os.path.join(npy_folder, npy_file)
        h5_path = os.path.join(h5ad_folder, h5_file)

        if not os.path.exists(npy_path):
            print(f"⚠️ Manca npy per {h5_file}, salto.")
            continue

        print(f"Caricamento {h5_file} + {npy_file}...")

        adata = sc.read_h5ad(h5_path)
        emb = np.load(npy_path)

        # Cell IDs
        if cell_id_in_h5ad is None:
            cell_ids = adata.obs_names.astype(str)
        else:
            if cell_id_in_h5ad not in adata.obs.columns:
                raise ValueError(f"Column `{cell_id_in_h5ad}` not found in adata.obs for {h5_file}")
            cell_ids = adata.obs[cell_id_in_h5ad].astype(str).values

        # Check dimensioni embedding
        if emb.shape[0] != len(cell_ids):
            raise ValueError(f"Numero di embedding ({emb.shape[0]}) diverso dal numero di cellule ({len(cell_ids)}) in {base_name}")

        # Allineamento label
        missing = [cid for cid in cell_ids if cid not in labels_df.index]
        if missing:
            print(f"⚠️ {len(missing)} cellule in {h5_file} senza label nel CSV. Imposto 'Unknown'.")
        labels_aligned = labels_df.reindex(cell_ids)[label_column_csv].fillna("Unknown").values

        dataset_dict = {
            "embedding": [e for e in emb],
            "label": labels_aligned.tolist(),
            "cell_id": cell_ids.tolist(),
            "source_file": [base_name] * len(cell_ids),
        }

        hf_ds = Dataset.from_dict(dataset_dict)
        all_datasets.append(hf_ds)

    full_dataset = concatenate_datasets(all_datasets)

    class_names = sorted(set(full_dataset["label"]))
    label_feature = ClassLabel(names=class_names)
    full_dataset = full_dataset.cast_column("label", label_feature)

    split_test = full_dataset.train_test_split(test_size=test_size, stratify_by_column="label", seed=random_state)
    split_val = split_test["train"].train_test_split(test_size=val_size, stratify_by_column="label", seed=random_state)

    dataset_splits = {
        "train": split_val["train"],
        "validation": split_val["test"],
        "test": split_test["test"],
    }

    print(f"Dataset creato con {len(full_dataset)} esempi, classi: {class_names}")

    return dataset_splits
