# Usage Guide - Mesh Processing Pipeline

## Overview

This guide provides detailed instructions for running the mesh normalization, quantization, and error analysis pipeline.

## File Descriptions

### Main Scripts

- **`mesh_analysis.ipynb`**: Interactive Jupyter notebook with step-by-step implementation
- **`mesh_complete_analysis.py`**: Complete standalone script that runs all tasks sequentially
- **`visualization_and_analysis.py`**: Focused script for generating visualizations and analysis

### Data Files

- **`data/sample_cube.obj`**: Simple 8-vertex cube mesh
- **`data/sample_sphere.obj`**: UV sphere with 722 vertices
- **`data/sample_torus.obj`**: Torus mesh with 1024 vertices

## Execution Methods

### Method 1: Complete Pipeline (Recommended for first run)

```bash
python mesh_complete_analysis.py
```

**What it does:**

1. Creates sample mesh files in `data/` directory
2. Loads and analyzes each mesh (statistics, properties)
3. Applies Min-Max and Unit Sphere normalization
4. Quantizes normalized coordinates using 1024 bins
5. Saves quantized meshes to `output/` directory
6. Reconstructs meshes through dequantization and denormalization
7. Calculates comprehensive error metrics
8. Saves reconstructed meshes
9. Generates analysis report with recommendations

**Expected Output:**

```
=== Mesh Normalization, Quantization, and Error Analysis ===
All essential libraries imported successfully!
Sample mesh files created in 'data/' directory

=== Analysis of data/sample_cube.obj ===
Number of vertices: 8
...processing continues...

PROJECT COMPLETED SUCCESSFULLY!
```

### Method 2: Interactive Notebook

```bash
jupyter notebook mesh_analysis.ipynb
```

**Step-by-step execution:**

1. **Cell 1**: Import libraries and verify installation
2. **Cell 2**: Create sample mesh files
3. **Cell 3**: Load and inspect meshes
4. **Cell 4**: Visualize original meshes
5. **Cell 5**: Define normalization classes
6. **Cell 6**: Process all meshes with both methods
7. **Cell 7**: Visualize normalization comparison
8. **Cell 8**: Perform reconstruction and error analysis
9. **Cell 9**: Generate error analysis plots
10. **Cell 10**: Create reconstruction comparison plots
11. **Cell 11**: Generate comprehensive analysis report
12. **Cell 12**: Create project documentation

### Method 3: Visualization Only

```bash
python visualization_and_analysis.py
```

**Use when:**

- You already have processed data
- You want to regenerate plots only
- You're experimenting with visualization parameters

## Understanding the Output

### Generated Files

**In `output/` directory:**

- `*_minmax_quantized.ply`: Meshes after Min-Max normalization and quantization
- `*_unitsphere_quantized.ply`: Meshes after Unit Sphere normalization and quantization
- `*_minmax_reconstructed.ply`: Reconstructed meshes from Min-Max method
- `*_unitsphere_reconstructed.ply`: Reconstructed meshes from Unit Sphere method

**In `visualizations/` directory:**

- `original_meshes.png`: 3D scatter plots of original mesh vertices
- `error_analysis.png`: Comprehensive error analysis charts

### Key Metrics Explained

**Mean Squared Error (MSE):**

- Measures average squared differences between original and reconstructed coordinates
- Lower values indicate better reconstruction quality
- Formula: `MSE = mean((original - reconstructed)²)`

**Mean Absolute Error (MAE):**

- Measures average absolute differences
- More interpretable than MSE for coordinate errors
- Formula: `MAE = mean(|original - reconstructed|)`

**Relative Error (%):**

- Error as percentage of original coordinate range
- Normalized measure for comparing different meshes
- Values < 1% indicate excellent quality preservation

## Customization Options

### Using Your Own Meshes

1. **Place your .obj files in the `data/` directory**
2. **Update the file list in the script:**
   ```python
   mesh_files = ['data/your_mesh1.obj', 'data/your_mesh2.obj']
   ```
3. **Run the analysis as normal**

### Adjusting Quantization

```python
# Change the number of quantization bins
n_bins = 512  # Default is 1024
n_bins = 2048  # For higher precision
```

### Adding New Normalization Methods

```python
def z_score_normalize(self, vertices, method_name='zscore'):
    """Z-score normalization example"""
    mean = vertices.mean(axis=0)
    std = vertices.std(axis=0)
    normalized = (vertices - mean) / std
    # Store normalization parameters...
```

## Performance Considerations

**Memory Usage:**

- Cube mesh: ~1KB memory
- Sphere mesh: ~50KB memory
- Torus mesh: ~80KB memory
- Total processing: <10MB RAM required

**Processing Time:**

- Complete analysis: ~30-60 seconds
- Visualization generation: ~10-20 seconds
- Depends on system specifications

## Troubleshooting Common Issues

### Issue: "No module named 'trimesh'"

**Solution:**

```bash
pip install trimesh
# or if that fails:
pip install --user trimesh
```

### Issue: Plots not showing in notebook

**Solution:**

```python
# Add this to notebook cell
%matplotlib inline
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode
```

### Issue: Permission denied when saving files

**Solution:**

```bash
# Ensure write permissions
chmod 755 output/
chmod 755 visualizations/
```

### Issue: Trimesh import warnings

**Solution:**

```python
import warnings
warnings.filterwarnings('ignore')
```

## Advanced Usage

### Batch Processing Multiple Meshes

```python
import glob
mesh_files = glob.glob('data/*.obj')
for filepath in mesh_files:
    # Process each mesh...
```

### Exporting Results to Different Formats

```python
# Export to different formats
mesh.export('output/result.stl')    # STL format
mesh.export('output/result.off')    # OFF format
mesh.export('output/result.dae')    # COLLADA format
```

### Creating Custom Visualizations

```python
# Custom color schemes
plt.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
           c=vertices[:, 0], cmap='plasma')

# Different plot styles
fig = plt.figure(figsize=(20, 10))  # Larger plots
ax.set_facecolor('black')           # Black background
```

## Research Applications

This pipeline is useful for:

- **AI Model Preparation**: Standardizing mesh data for neural networks
- **3D Graphics Research**: Understanding quantization effects on geometry
- **Data Compression**: Analyzing information loss in coordinate discretization
- **Quality Assessment**: Benchmarking different preprocessing methods
- **SeamGPT-style Systems**: Preparing 3D data for AI understanding

## Next Steps

After running this analysis, consider:

1. Testing with more complex mesh geometries
2. Implementing adaptive quantization based on local mesh density
3. Adding rotation and translation invariance testing
4. Exploring hierarchical mesh decomposition
5. Creating web-based visualization interfaces
