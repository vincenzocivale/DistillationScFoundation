
import scanpy as sc
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from distilled_tx1.preprocessing.pipeline import TahoePreprocessor, PreprocessingConfig
from distilled_tx1.models.modeling_distilled_tahoe import DistilledTahoeModel
import torch
import os
import shutil

# --- Configuration ---
# IMPORTANT: Replace with your Hugging Face username and desired model name
HF_USERNAME = "Yuto2007"
MODEL_NAME = "distilled-tahoe-sc-foundation"
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

# Paths
MODEL_DIR = Path("./model_outputs/distilled_tahoe")
PREPROCESSOR_DIR = Path("./model_outputs/preprocessor")
TEMP_UPLOAD_DIR = Path("./temp_upload")

# --- Main Script ---
def main():
    """
    This script prepares your trained model and preprocessor and uploads them
    to the Hugging Face Hub.
    """
    print("--- Preparing for Hugging Face Hub Upload ---")

    # 1. Create a temporary directory for all artifacts
    if TEMP_UPLOAD_DIR.exists():
        shutil.rmtree(TEMP_UPLOAD_DIR)
    TEMP_UPLOAD_DIR.mkdir(parents=True)
    print(f"Created temporary upload directory: {TEMP_UPLOAD_DIR}")

    # 2. Load and save the preprocessor
    print("\nStep 1: Preparing the Preprocessor")
    
    # Use the same config as in your training script
    preprocessor_config = PreprocessingConfig(
        seq_len=512,
        n_bins=51,
        normalize=False,
        target_sum=1e4,
        gene_sampling_strategy="topk",
        add_cls_token=True,
        gene_id_key="gene_id"
    )
    
    preprocessor = TahoePreprocessor(
        config=preprocessor_config,
        vocab_path="vocab.json"
    )

    # The binner needs to be fitted. We'll process a small adata file to do this.
    # This ensures the binner statistics are computed and saved.
    try:
        ref_adata = sc.read_h5ad("data_yuto_with_clusters_chunk_001.h5ad")
        print("Fitting the preprocessor's binner...")
        # We don't need the output, just the side effect of fitting the binner
        _ = preprocessor.process_adata(ref_adata, return_dict=True)
    except FileNotFoundError:
        print("\nWARNING: Could not find 'data_yuto_with_clusters_chunk_001.h5ad'.")
        print("The preprocessor's binner will not be fitted. This may cause issues during inference.")
        print("Please ensure a sample h5ad file is available to fit the binner, or fit it manually.")
    except Exception as e:
        print(f"\nAn error occurred while fitting the binner: {e}")
        print("Please resolve the issue and try again.")
        return

    # Save the fitted preprocessor to the temporary directory
    preprocessor.save(TEMP_UPLOAD_DIR)
    print(f"Preprocessor saved to: {TEMP_UPLOAD_DIR}")

    # 3. Copy the trained model files
    print("\nStep 2: Preparing the Model")
    if not MODEL_DIR.exists():
        print(f"ERROR: Model directory not found at {MODEL_DIR}")
        print("Please ensure your model has been trained and the files are in the correct location.")
        return

    for item in os.listdir(MODEL_DIR):
        s = os.path.join(MODEL_DIR, item)
        d = os.path.join(TEMP_UPLOAD_DIR, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks=True)
        else:
            shutil.copy2(s, d)
            
    print(f"Model files copied from {MODEL_DIR} to {TEMP_UPLOAD_DIR}")


    # 4. Upload to Hugging Face Hub
    print(f"\nStep 3: Uploading to Hugging Face Hub (Repo: {REPO_ID})")
    print("This requires you to be logged in via `huggingface-cli login`.")
    
    try:
        # Create the repository (if it doesn't exist)
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print(f"Successfully created or found repo: {REPO_ID}")

        # Upload the entire contents of the temporary directory
        api = HfApi()
        api.upload_folder(
            folder_path=TEMP_UPLOAD_DIR,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Upload distilled student model and preprocessor"
        )
        print("\n--- Upload Complete! ---")
        print(f"Your model is now available at: https://huggingface.co/{REPO_ID}")

    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please check your Hugging Face credentials and network connection.")
    finally:
        # Clean up the temporary directory
        print(f"\nCleaning up temporary directory: {TEMP_UPLOAD_DIR}")
        shutil.rmtree(TEMP_UPLOAD_DIR)


if __name__ == "__main__":
    # Before running, make sure to log in to Hugging Face Hub:
    # In your terminal, run: huggingface-cli login
    # You'll need to provide a token with 'write' access.
    
    if HF_USERNAME == "your-username":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE EDIT THE SCRIPT TO SET YOUR HF_USERNAME !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        main()
