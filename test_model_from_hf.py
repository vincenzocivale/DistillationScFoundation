
import scanpy as sc
from pathlib import Path
from huggingface_hub import hf_hub_download
from distilled_tx1.preprocessing.pipeline import TahoePreprocessor
from distilled_tx1.models.modeling_distilled_tahoe import DistilledTahoeModel
import torch
import os
import matplotlib.pyplot as plt
from anndata import AnnData

# --- Configuration ---
# IMPORTANT: Replace with the REPO_ID of your model on the Hugging Face Hub
HF_USERNAME = "Yuto2007"
MODEL_NAME = "distilled-tahoe-sc-foundation"
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

# File to use for testing inference
# IMPORTANT: Replace with the path to your test data file
TEST_DATA_H5AD = "/data/scClassificationDatasets/data_yuto/processed_tahoe_x1/data_yuto_with_clusters_chunk_020.h5ad"

# --- Main Script ---
def main():
    """
    This script downloads your model and preprocessor from the Hugging Face Hub,
    runs inference on a test file, and saves the results.
    """
    print(f"--- Testing Model from Hugging Face Hub: {REPO_ID} ---")

    # 1. Download and load the preprocessor
    print("\nStep 1: Loading Preprocessor from Hub")
    preprocessor_dir = Path("./model_outputs/preprocessor_from_hf")
    preprocessor_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download the necessary files for the preprocessor
        for filename in ["preprocessing_config.json", "vocab.json", "binner.json"]:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=preprocessor_dir,
                local_dir_use_symlinks=False, # Use False to avoid symlink issues
                force_download=True # Force download to get the latest version
            )
        
        preprocessor = TahoePreprocessor.load(preprocessor_dir)
        print("Preprocessor loaded successfully.")

    except Exception as e:
        print(f"\nAn error occurred while loading the preprocessor: {e}")
        print("Please check that the REPO_ID is correct and the files exist on the Hub.")
        return

    # 2. Load the model from the Hub
    print("\nStep 2: Loading Model from Hub")
    try:
        model = DistilledTahoeModel.from_pretrained(REPO_ID)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"\nAn error occurred while loading the model: {e}")
        print("Please check that the REPO_ID is correct and you have the necessary libraries installed.")
        return

    # 3. Load test data
    print("\nStep 3: Loading Test Data")
    test_file = Path(TEST_DATA_H5AD)
    if not test_file.exists():
        print(f"ERROR: Test data file not found at: {TEST_DATA_H5AD}")
        print("Please update the TEST_DATA_H5AD variable in the script.")
        return
    
    adata = sc.read_h5ad(test_file)
    print(f"Loaded test data with {adata.n_obs} cells and {adata.n_vars} genes.")

    # 4. Run preprocessing and inference
    print("\nStep 4: Running Preprocessing and Inference")
    with torch.no_grad():
        # Preprocess the data
        # Note: The preprocessor expects the gene_id_key in adata.var
        # If your test data has a different column name for gene IDs, you might need to adjust it
        # For this example, we assume it's the same as in training ('gene_id')
        
        # We need to make sure the test adata has the same var names as the training
        # A simple way is to reindex based on the vocabulary
        # This is a simplified approach; a more robust pipeline might be needed for production
        vocab_genes = list(preprocessor.vocab.gene_to_idx.keys())
        
        # Make sure we don't have duplicate gene IDs in the anndata object
        adata.var.index = adata.var[preprocessor.config.gene_id_key]
        adata = adata[:, ~adata.var.index.duplicated(keep='first')]

        shared_genes = [gene for gene in vocab_genes if gene in adata.var_names]
        
        adata = adata[:, shared_genes].copy()
        
        print(f"Aligned test data to vocabulary. Found {len(shared_genes)} shared genes.")

        processed_input = preprocessor.process_adata(adata, return_dict=True)
        
        # Move tensors to the correct device (e.g., 'cuda' if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        input_ids = processed_input["gene_ids"].to(device)
        expression_bins = processed_input["expression_bins"].to(device)
        attention_mask = processed_input["attention_mask"].to(device)

        # Get cell embeddings
        embeddings = model.get_cell_embeddings(
            gene_ids=input_ids,
            expression_bins=expression_bins,
            attention_mask=attention_mask
        )
        
        embeddings_np = embeddings.cpu().numpy()

    print(f"Inference complete. Generated embeddings of shape: {embeddings_np.shape}")

    # 5. Save results
    print("\nStep 5: Saving Results")
    adata.obsm['distilled_embedding'] = embeddings_np
    
    output_filename = test_file.stem + "_with_embeddings.h5ad"
    adata.write_h5ad(output_filename)
    print(f"Results saved to: {output_filename}")
    print("\n--- Test Complete! ---")

    # 6. Generate and save UMAP plots for visual comparison
    print("\nStep 6: Generating UMAP Comparison Plots")
    try:
        # We need the teacher embeddings for comparison.
        # Assuming they are in the same AnnData object, loaded from the test file.
        if 'Tx1-70m' in adata.obsm:
            teacher_embeddings = adata.obsm['Tx1-70m']
            
            # Ensure teacher and student embeddings have the same number of observations
            if teacher_embeddings.shape[0] == embeddings_np.shape[0]:
                # Create a temporary AnnData for plotting
                plot_adata = AnnData(X=adata.X, obs=adata.obs.copy())
                plot_adata.obsm['student'] = embeddings_np
                plot_adata.obsm['teacher'] = teacher_embeddings
                
                # It's good practice to have a color key if available, e.g., 'cell_type'
                color_key = None
                if 'cell_type' in plot_adata.obs:
                    color_key = 'cell_type'
                elif 'cluster' in plot_adata.obs:
                    color_key = 'cluster'

                # Generate UMAP for student embeddings
                sc.pp.neighbors(plot_adata, use_rep='student', n_neighbors=15)
                sc.tl.umap(plot_adata)
                sc.pl.umap(plot_adata, color=color_key, title="Student Embeddings", save="_student_embeddings.png", show=False)
                
                # Generate UMAP for teacher embeddings
                sc.pp.neighbors(plot_adata, use_rep='teacher', n_neighbors=15)
                sc.tl.umap(plot_adata)
                sc.pl.umap(plot_adata, color=color_key, title="Teacher Embeddings", save="_teacher_embeddings.png", show=False)

                print("UMAP plots saved to 'figures' directory (e.g., 'figures/umap_student_embeddings.png').")

            else:
                print("Warning: Mismatch in the number of cells between teacher and student embeddings. Skipping UMAP plots.")
        else:
            print("Warning: 'Tx1-70m' (teacher embeddings) not found in the test data. Skipping UMAP comparison.")
            
    except Exception as e:
        print(f"An error occurred during UMAP plot generation: {e}")

if __name__ == "__main__":
    if HF_USERNAME == "your-username":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE EDIT THE SCRIPT TO SET YOUR HF_USERNAME !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    elif TEST_DATA_H5AD == "path/to/your/test_data.h5ad":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE EDIT THE SCRIPT TO SET THE TEST_DATA_H5AD PATH !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        main()
