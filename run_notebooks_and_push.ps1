# Execute all notebooks in-place, display saved visualization images inside notebooks, then commit and push changes to Git

# Requirements: Python with jupyter installed and git configured for the repo

$root = "c:\Users\User\Documents\Online Projects\Ai\Machine Learning\Construction models"
cd $root

# List of notebooks to execute
$notebooks = @( 
    "archi_high_accuracy_model\Architectural_Analysis_Notebook.ipynb",
    "Mepfs_high_accuracy_model\Mepfs_workflow.ipynb",
    "Struc_high_accuracy_model\Structural_Analysis_Notebook.ipynb"
)

# Ensure jupyter nbconvert is available
Write-Host "Checking for jupyter nbconvert..."
python -c "import nbformat, nbconvert; print('nbconvert OK')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing nbconvert..."
    pip install nbconvert --quiet
}

# Execute each notebook in-place
foreach ($nb in $notebooks) {
    $path = Join-Path $root $nb
    Write-Host "Executing: $path"
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 "$path"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Execution failed for $path"
        exit 1
    }
}

# Stage and commit the executed notebooks
git add $notebooks
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
git commit -m "Execute notebooks and save outputs: $timestamp" || Write-Host "No changes to commit"

# Push to the current branch
git push

Write-Host "Done. Notebooks executed, saved, and pushed."