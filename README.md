# Mesh Normalization, Quantization, and Error Analysis

## Project Overview

This project implements comprehensive 3D mesh preprocessing techniques for AI model preparation, focusing on understanding and implementing data preprocessing for 3D meshes - a fundamental step in research workflows for systems like SeamGPT.

## 🎯 Project Goals

- Implement and compare different mesh normalization techniques
- Apply quantization to reduce coordinate precision while preserving structure
- Analyze reconstruction quality and information loss
- Generate comprehensive visualizations and error analysis reports

## Project Structure

```
mesh_assignment/
├── mesh_analysis.ipynb              # Jupyter notebook implementation
├── mesh_complete_analysis.py        # Complete standalone Python script
├── visualization_and_analysis.py    # Visualization and analysis script
├── README.md                        # This file
├── data/                            # Input mesh files
│   ├── sample_cube.obj              # Sample cube mesh
│   ├── sample_sphere.obj            # Sample sphere mesh
│   └── sample_torus.obj             # Sample torus mesh
├── output/                          # Processed mesh files
│   ├── sample_cube_minmax_quantized.ply
│   ├── sample_cube_unitsphere_quantized.ply
│   ├── sample_cube_minmax_reconstructed.ply
│   ├── sample_cube_unitsphere_reconstructed.ply
│   ├── sample_sphere_minmax_quantized.ply
│   ├── sample_sphere_unitsphere_quantized.ply
│   ├── sample_sphere_minmax_reconstructed.ply
│   ├── sample_sphere_unitsphere_reconstructed.ply
│   ├── sample_torus_minmax_quantized.ply
│   ├── sample_torus_unitsphere_quantized.ply
│   ├── sample_torus_minmax_reconstructed.ply
│   └── sample_torus_unitsphere_reconstructed.ply
└── visualizations/                  # Generated plots and images
    ├── original_meshes.png          # Original mesh visualizations
    └── error_analysis.png           # Error analysis plots
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.7+** (Tested on Python 3.13)
- **Git** (for cloning the repository)
- **Jupyter Notebook** or **VS Code** (for notebook execution)

### Installation

1. **Clone or download the project:**

   ```bash
   git clone [https://github.com/gokulkumarv24/ml-3d](https://github.com/gokulkumarv24/ml-3d)
   cd mesh_assignment
   ```
2. **Install required packages:**

   ```bash
   pip install numpy matplotlib trimesh pandas scikit-learn
   ```

   Or install all at once:

   ```bash
   pip install -r requirements.txt
   ```
3. **Verify installation:**

   ```bash
   python -c "import numpy, matplotlib, trimesh, pandas, sklearn; print('All packages installed successfully!')"
   ```

### 📋 Execution Options

#### Option 1: Complete Analysis (Recommended)

Run the full analysis pipeline with all tasks:

```bash
python mesh_complete_analysis.py
```

**Expected output:**

- Creates sample mesh files
- Processes all meshes with both normalization methods
- Generates quantized and reconstructed meshes
- Creates visualizations (saved to `visualizations/`)
- Displays comprehensive analysis report

#### Option 2: Interactive Jupyter Notebook

For step-by-step execution and experimentation:

```bash
# Start notebook server
jupyter notebook mesh_analysis.ipynb

# Or if using VS Code
code mesh_analysis.ipynb
```

**Steps to run:**

1. Execute cells sequentially (Shift+Enter)
2. First cell: Import libraries
3. Second cell: Create sample meshes
4. Continue through all cells for complete analysis

#### Option 3: Visualization and Analysis Only

If you already have processed data:

```bash
python visualization_and_analysis.py
```

#### Option 4: Custom Mesh Analysis

To analyze your own mesh files:

```python
# Replace sample mesh files in data/ folder with your .obj files
# Update file paths in the script
mesh_files = ['data/your_mesh.obj']
```

### 🔧 Troubleshooting

**Common Issues:**

1. **Import Error for trimesh:**

   ```bash
   pip install --upgrade trimesh
   ```
2. **matplotlib display issues:**

   ```bash
   # For headless systems
   export MPLBACKEND=Agg
   ```
3. **Permission errors on Windows:**

   ```bash
   # Run as administrator or use
   pip install --user numpy matplotlib trimesh pandas scikit-learn
   ```
4. **Jupyter notebook not starting:**

   ```bash
   pip install jupyter
   jupyter notebook --generate-config
   ```

## Implementation Details

### Task 1: Load and Inspect Mesh Data

**Implemented Features:**

- Created sample mesh files (cube, sphere, torus) in .obj format
- Loaded meshes using trimesh library
- Extracted vertex coordinates as NumPy arrays
- Computed comprehensive statistics per axis:
  - Number of vertices and faces
  - Min, max, mean, standard deviation
  - Range and centroid calculations
  - Bounding box volume, surface area, and volume
- Generated 3D visualizations of original meshes

**Sample Output:**

```
=== Analysis of data/sample_cube.obj ===
Number of vertices: 8
Number of faces: 12
Vertex Statistics:
Axis     Min    Max   Mean    Std  Range
   X -1.0000 1.0000 0.0000 1.0000 2.0000
   Y -1.0000 1.0000 0.0000 1.0000 2.0000
   Z -1.0000 1.0000 0.0000 1.0000 2.0000
```

### Task 2: Normalize and Quantize Mesh

**Implemented Normalization Methods:**

1. **Min-Max Normalization**

   - Formula: `x' = (x - x_min) / (x_max - x_min)`
   - Brings coordinates to [0, 1] range
   - Preserves original aspect ratios
2. **Unit Sphere Normalization**

   - Centers mesh at origin: `centered = vertices - centroid`
   - Scales to fit in unit sphere: `normalized = centered / max_distance`
   - Results in [-1, 1] range for most coordinates

**Quantization Process:**

- Used 1024 bins for discretization
- Formula: `q = floor(x' × (n_bins - 1))`
- Handled coordinate range shifting for proper quantization
- Saved quantized meshes in PLY format

**Results:**

- Generated 6 quantized mesh files (2 methods × 3 meshes)
- Achieved efficient bin utilization (2-251 bins used per mesh)
- Preserved mesh topology while reducing precision

### Task 3: Reconstruction and Error Analysis

**Reconstruction Process:**

1. **Dequantization**: `x'' = q / (n_bins - 1)`
2. **Denormalization**: Applied inverse of normalization transforms
3. **Error Calculation**: Computed MSE and MAE between original and reconstructed

**Error Metrics Implemented:**

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Per-axis error breakdown
- Relative error percentage
- Statistical comparisons between methods

**Key Results:**

| Mesh   | Best Method | MSE      | Relative Error |
| ------ | ----------- | -------- | -------------- |
| Cube   | Min-Max     | 0.000000 | 0.0%           |
| Sphere | Min-Max     | 0.000003 | <0.06%         |
| Torus  | Min-Max     | 0.000005 | <0.28%         |

## Analysis Results

### Key Findings

1. **Best Performing Method**: **Min-Max Normalization**

   - Average MSE: 0.00000279
   - Consistently outperformed Unit Sphere normalization
   - Perfect reconstruction for simple geometries (cube)
2. **Quantization Effectiveness**

   - 1024 bins provide excellent quality preservation (<1% relative error)
   - Systematic errors depend on normalization method choice
   - Complex geometries show higher reconstruction errors
3. **Method Comparison**

   - **Min-Max**: Better for preserving original aspect ratios
   - **Unit Sphere**: More uniform scaling, better for certain applications
   - **Geometry Impact**: Simple shapes reconstruct perfectly with Min-Max

### Conclusions

- **Min-Max normalization** generally produces lower reconstruction errors
- Preserving original aspect ratios is beneficial for most mesh types
- **Quantization with 1024 bins** preserves mesh structure very well
- Complex geometries (torus) show different error patterns than simple shapes
- Choice of normalization method should consider specific application requirements

### Recommendations

1. Use **Min-Max normalization** for best overall accuracy in this dataset
2. Consider **Min-Max** for geometries with clear bounding boxes
3. Consider **Unit Sphere** for shapes requiring uniform scaling
4. Increase quantization bins if higher precision is needed
5. Validate results with additional mesh types and sizes

## Visualizations Generated

1. **Original Meshes** (`visualizations/original_meshes.png`)

   - 3D scatter plots of all three sample meshes
   - Color-coded by Z-coordinate values
   - Equal axis scaling for proper proportions
2. **Error Analysis** (`visualizations/error_analysis.png`)

   - MSE and MAE comparisons between methods
   - Per-axis relative error breakdown
   - Statistical summary tables
   - Comprehensive performance comparison

## Technical Implementation

### Core Classes and Functions

- `MeshNormalizer`: Handles both normalization methods with parameter storage
- `quantize_vertices()`: Converts continuous coordinates to discrete bins
- `dequantize_vertices()`: Recovers continuous values from quantized data
- `load_and_inspect_mesh()`: Complete mesh analysis and statistics
- Visualization functions for comprehensive result presentation

### Error Handling

- Division by zero protection in normalization
- Coordinate range validation for quantization
- Robust parameter storage and retrieval system
- Comprehensive error metrics calculation

## 📊 Project Completion Status

- **Task 1**: ✅ Complete mesh loading, inspection, and visualization
- **Task 2**: ✅ Two normalization methods with quantization implementation
- **Task 3**: ✅ Full reconstruction pipeline with comprehensive error analysis
- **Documentation**: ✅ Complete README, code comments, and analysis report
- **Deliverables**: ✅ All required files, plots, and processed meshes generated

**Implementation Status**: Complete with comprehensive analysis and documentation

## 🎓 Educational Value

This project demonstrates:

- **Data Preprocessing**: Normalization and quantization techniques
- **3D Graphics**: Mesh data structures and coordinate transformations
- **Error Analysis**: Statistical methods for quality assessment
- **Visualization**: Professional scientific plotting and presentation
- **Software Engineering**: Clean, documented, and reusable code

## 🤝 Contributing

To extend this project:

1. Add new normalization methods in the `MeshNormalizer` class
2. Implement adaptive quantization techniques
3. Add support for different mesh file formats
4. Create web-based visualization interface
5. Optimize for large-scale mesh processing

## 📚 References

- Trimesh documentation: https://trimsh.org/
- NumPy user guide: https://numpy.org/doc/stable/
- Matplotlib tutorials: https://matplotlib.org/stable/tutorials/index.html
- 3D mesh processing fundamentals: Computer Graphics literature

## Files Description

- **Scripts**: Complete Python implementations with detailed documentation
- **Data**: Sample mesh files in OBJ format for testing
- **Output**: All processed meshes in PLY format for further analysis
- **Visualizations**: High-quality plots for result presentation and analysis
- **Documentation**: Comprehensive README and inline code documentation

This implementation provides a solid foundation for understanding 3D mesh preprocessing techniques essential for AI model preparation in graphics and computer vision applications.
