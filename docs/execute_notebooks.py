"""
    Script for updating notebooks on documentation branch.
"""
import papermill as pm
import pathlib
from aurora.general_helper_functions import replace_in_file
from aurora.general_helper_functions import DOCS_PATH


# Define the root directory for your notebooks
notebook_dir = DOCS_PATH.joinpath("examples")

# Recursively find all .ipynb files in the directory
notebooks = sorted(
    nb for nb in notebook_dir.rglob("*.ipynb") if ".ipynb_checkpoints" not in str(nb)
)
notebook_dir = DOCS_PATH.joinpath("tutorials")
notebooks += sorted(
    nb for nb in notebook_dir.rglob("*.ipynb") if ".ipynb_checkpoints" not in str(nb)
)   

# Execute each notebook in-place
for nb_path in notebooks:
    nb_path = nb_path.resolve()
    working_dir = nb_path.parent
    replace_in_file(
        nb_path,
        "%matplotlib widget",
        "%matplotlib inline",
    )
    print(f"Executing: {nb_path} (in cwd={working_dir})")
    
    try:
        pm.execute_notebook(
            input_path=str(nb_path),
            output_path=str(nb_path),
            kernel_name="aurora-test",     # Adjust if using a different kernel ("dipole-st")
            request_save_on_cell_execute=True,
            cwd=str(working_dir)  # <- this sets the working directory!
        )
        print(f"✓ Executed successfully: {nb_path}")
    except Exception as e:
        print(f"✗ Failed to execute {nb_path}: {e}")
        exit(1) 

    # Replace the matplotlib inline magic back to widget for interactive plots
    replace_in_file(
        nb_path,
        "%matplotlib inline",
        "%matplotlib widget",
    )
print("All notebooks executed and updated successfully.")

