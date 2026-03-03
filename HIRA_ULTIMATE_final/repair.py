import h5py
import os

# Define the folder containing your 6 models
MODEL_FOLDER = r'C:\Users\mazin\Downloads\HIRA_ULTIMATE\ai_models'

def repair_model(file_path):
    """Removes the 'groups': 1 keyword from an H5 model's config."""
    try:
        with h5py.File(file_path, mode="r+") as f:
            # Check if model_config attribute exists
            if 'model_config' in f.attrs:
                config_str = f.attrs.get("model_config")
                
                # Check for the problematic keyword
                if '"groups": 1,' in config_str:
                    new_config = config_str.replace('"groups": 1,', '')
                    f.attrs.modify('model_config', new_config)
                    f.flush()
                    print(f"✅ Successfully repaired: {os.path.basename(file_path)}")
                else:
                    print(f"ℹ️ No 'groups' keyword found in: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"❌ Failed to repair {os.path.basename(file_path)}: {e}")

# Walk through all subfolders (xray, mri, ct, etc.)
for root, dirs, files in os.walk(MODEL_FOLDER):
    for file in files:
        if file.endswith(".h5"):
            repair_model(os.path.join(root, file))