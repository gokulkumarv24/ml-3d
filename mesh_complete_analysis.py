# Mesh Normalization, Quantization, and Error Analysis
# Complete assignment implementation

import numpy as np
import matplotlib.pyplot as plt
import trimesh
import pandas as pd
import os
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')
plt.style.use('default')

print("=== Mesh Normalization, Quantization, and Error Analysis ===")
print("All essential libraries imported successfully!")
print(f"NumPy version: {np.__version__}")
print(f"Trimesh version: {trimesh.__version__}")

# Create sample mesh files for demonstration
def create_sample_meshes():
    """Create sample mesh files for testing"""
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Create a simple cube mesh
    cube = trimesh.creation.box(extents=[2, 2, 2])
    cube.export('data/sample_cube.obj')
    
    # Create a sphere mesh
    sphere = trimesh.creation.uv_sphere(radius=1.5, count=[20, 20])
    sphere.export('data/sample_sphere.obj')
    
    # Create a more complex mesh - torus
    torus = trimesh.creation.torus(major_radius=2, minor_radius=0.5)
    torus.export('data/sample_torus.obj')
    
    print("Sample mesh files created in 'data/' directory")
    return ['data/sample_cube.obj', 'data/sample_sphere.obj', 'data/sample_torus.obj']

# Create sample meshes
mesh_files = create_sample_meshes()

# Task 1: Load and Inspect the Mesh
def load_and_inspect_mesh(filepath):
    """Load mesh and extract basic statistics"""
    
    print(f"\\n=== Analysis of {filepath} ===")
    
    # Load mesh using trimesh
    mesh = trimesh.load(filepath)
    vertices = mesh.vertices
    
    # Extract vertex coordinates
    print(f"Mesh loaded successfully!")
    print(f"Number of vertices: {len(vertices)}")
    print(f"Number of faces: {len(mesh.faces)}")
    print(f"Vertex array shape: {vertices.shape}")
    
    # Compute statistics for each axis
    stats_df = pd.DataFrame({
        'Axis': ['X', 'Y', 'Z'],
        'Min': vertices.min(axis=0),
        'Max': vertices.max(axis=0),
        'Mean': vertices.mean(axis=0),
        'Std': vertices.std(axis=0),
        'Range': vertices.max(axis=0) - vertices.min(axis=0)
    })
    
    print("\\nVertex Statistics:")
    print(stats_df.to_string(index=False, float_format='%.4f'))
    
    # Overall mesh properties
    centroid = vertices.mean(axis=0)
    bounding_box_volume = np.prod(vertices.max(axis=0) - vertices.min(axis=0))
    
    print(f"\\nMesh Properties:")
    print(f"Centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")
    print(f"Bounding box volume: {bounding_box_volume:.4f}")
    print(f"Surface area: {mesh.area:.4f}")
    print(f"Volume: {mesh.volume:.4f}")
    
    return mesh, vertices, stats_df

# Load and inspect all sample meshes
meshes_data = {}
for filepath in mesh_files:
    mesh_name = os.path.basename(filepath).split('.')[0]
    mesh, vertices, stats = load_and_inspect_mesh(filepath)
    meshes_data[mesh_name] = {
        'mesh': mesh,
        'vertices': vertices,
        'stats': stats
    }

# Task 2: Normalize and Quantize the Mesh
class MeshNormalizer:
    """Class for different mesh normalization methods"""
    
    def __init__(self):
        self.normalization_params = {}
    
    def min_max_normalize(self, vertices, method_name='minmax'):
        """Min-Max normalization to [0, 1] range"""
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        
        # Avoid division by zero
        range_vals = v_max - v_min
        range_vals[range_vals == 0] = 1
        
        normalized = (vertices - v_min) / range_vals
        
        # Store parameters for denormalization
        self.normalization_params[method_name] = {
            'type': 'minmax',
            'min': v_min,
            'max': v_max,
            'range': range_vals
        }
        
        return normalized
    
    def unit_sphere_normalize(self, vertices, method_name='unitsphere'):
        """Unit sphere normalization - fit mesh within unit sphere"""
        # Center the mesh at origin
        centroid = vertices.mean(axis=0)
        centered = vertices - centroid
        
        # Find maximum distance from center
        max_distance = np.linalg.norm(centered, axis=1).max()
        
        # Avoid division by zero
        if max_distance == 0:
            max_distance = 1
        
        # Scale to fit in unit sphere
        normalized = centered / max_distance
        
        # Store parameters for denormalization
        self.normalization_params[method_name] = {
            'type': 'unitsphere',
            'centroid': centroid,
            'max_distance': max_distance
        }
        
        return normalized
    
    def denormalize(self, normalized_vertices, method_name):
        """Reverse the normalization process"""
        params = self.normalization_params[method_name]
        
        if params['type'] == 'minmax':
            return normalized_vertices * params['range'] + params['min']
        
        elif params['type'] == 'unitsphere':
            return normalized_vertices * params['max_distance'] + params['centroid']
        
        else:
            raise ValueError(f"Unknown normalization type: {params['type']}")

def quantize_vertices(normalized_vertices, n_bins=1024):
    """Quantize normalized vertices to discrete bins"""
    # Ensure vertices are in [0, 1] range for quantization
    # If they're in [-1, 1], shift to [0, 1]
    if normalized_vertices.min() < 0:
        shifted = (normalized_vertices + 1) / 2
        was_shifted = True
    else:
        shifted = normalized_vertices
        was_shifted = False
    
    # Quantize
    quantized = np.floor(shifted * (n_bins - 1)).astype(int)
    
    # Ensure values are within valid range
    quantized = np.clip(quantized, 0, n_bins - 1)
    
    return quantized, was_shifted

def dequantize_vertices(quantized_vertices, n_bins=1024, was_shifted=False):
    """Dequantize vertices back to continuous values"""
    dequantized = quantized_vertices / (n_bins - 1)
    
    # If vertices were shifted during quantization, shift back
    if was_shifted:
        dequantized = dequantized * 2 - 1
    
    return dequantized

print("\\nNormalization and quantization classes defined successfully!")

# Apply normalization and quantization to all meshes
normalizer = MeshNormalizer()
n_bins = 1024

processed_meshes = {}

for mesh_name, data in meshes_data.items():
    print(f"\\n=== Processing {mesh_name} ===")
    
    vertices = data['vertices']
    mesh = data['mesh']
    
    # Apply both normalization methods
    methods = {
        'minmax': normalizer.min_max_normalize,
        'unitsphere': normalizer.unit_sphere_normalize
    }
    
    processed_meshes[mesh_name] = {'original': vertices}
    
    for method_name, normalize_func in methods.items():
        print(f"\\nApplying {method_name} normalization...")
        
        # Normalize
        normalized = normalize_func(vertices, f"{mesh_name}_{method_name}")
        
        print(f"Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
        
        # Quantize
        quantized, was_shifted = quantize_vertices(normalized, n_bins)
        
        print(f"Quantized range: [{quantized.min()}, {quantized.max()}]")
        print(f"Quantization bins used: {len(np.unique(quantized.flatten()))}")
        
        # Store results
        processed_meshes[mesh_name][method_name] = {
            'normalized': normalized,
            'quantized': quantized,
            'was_shifted': was_shifted
        }
        
        # Save quantized mesh
        quantized_mesh = mesh.copy()
        quantized_mesh.vertices = quantized.astype(float)  # Convert to float for saving
        
        output_path = f"output/{mesh_name}_{method_name}_quantized.ply"
        quantized_mesh.export(output_path)
        print(f"Saved quantized mesh to: {output_path}")

print("\\nAll meshes processed successfully!")

# Task 3: Dequantize, Denormalize, and Measure Error
reconstruction_results = {}

for mesh_name, data in processed_meshes.items():
    if mesh_name not in reconstruction_results:
        reconstruction_results[mesh_name] = {}
    
    original = data['original']
    
    for method in ['minmax', 'unitsphere']:
        print(f"\\n=== Reconstructing {mesh_name} with {method} ===")
        
        # Get processed data
        quantized = data[method]['quantized']
        was_shifted = data[method]['was_shifted']
        
        # Dequantize
        dequantized = dequantize_vertices(quantized, n_bins, was_shifted)
        print(f"Dequantized range: [{dequantized.min():.4f}, {dequantized.max():.4f}]")
        
        # Denormalize
        reconstructed = normalizer.denormalize(dequantized, f"{mesh_name}_{method}")
        print(f"Reconstructed range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
        
        # Calculate errors
        mse = mean_squared_error(original, reconstructed)
        mae = mean_absolute_error(original, reconstructed)
        
        # Per-axis errors
        mse_per_axis = np.mean((original - reconstructed) ** 2, axis=0)
        mae_per_axis = np.mean(np.abs(original - reconstructed), axis=0)
        
        # Relative error
        original_range = original.max(axis=0) - original.min(axis=0)
        relative_error = np.sqrt(mse_per_axis) / original_range * 100
        
        print(f"Mean Squared Error (MSE): {mse:.8f}")
        print(f"Mean Absolute Error (MAE): {mae:.8f}")
        print(f"MSE per axis (X, Y, Z): [{mse_per_axis[0]:.8f}, {mse_per_axis[1]:.8f}, {mse_per_axis[2]:.8f}]")
        print(f"Relative error per axis (%): [{relative_error[0]:.4f}, {relative_error[1]:.4f}, {relative_error[2]:.4f}]")
        
        # Store results
        reconstruction_results[mesh_name][method] = {
            'reconstructed': reconstructed,
            'mse': mse,
            'mae': mae,
            'mse_per_axis': mse_per_axis,
            'mae_per_axis': mae_per_axis,
            'relative_error': relative_error
        }
        
        # Save reconstructed mesh
        reconstructed_mesh = trimesh.Trimesh(vertices=reconstructed, 
                                            faces=meshes_data[mesh_name]['mesh'].faces)
        output_path = f"output/{mesh_name}_{method}_reconstructed.ply"
        reconstructed_mesh.export(output_path)
        print(f"Saved reconstructed mesh to: {output_path}")

print("\\nReconstruction completed for all meshes!")

# Create visualizations
def visualize_original_meshes(meshes_data):
    """Visualize original meshes using matplotlib"""
    
    fig = plt.figure(figsize=(15, 5))
    
    for i, (name, data) in enumerate(meshes_data.items(), 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        
        vertices = data['vertices']
        
        # Plot vertices as scatter plot
        scatter = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                           c=vertices[:, 2], cmap='viridis', s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Original {name.replace("_", " ").title()}')
        
        # Make axes equal
        max_range = np.array([vertices[:,0].max()-vertices[:,0].min(),
                             vertices[:,1].max()-vertices[:,1].min(),
                             vertices[:,2].max()-vertices[:,2].min()]).max() / 2.0
        
        mid_x = (vertices[:,0].max()+vertices[:,0].min()) * 0.5
        mid_y = (vertices[:,1].max()+vertices[:,1].min()) * 0.5
        mid_z = (vertices[:,2].max()+vertices[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig('visualizations/original_meshes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Original meshes visualization saved to visualizations/original_meshes.png")

# Visualize original meshes
visualize_original_meshes(meshes_data)

# Create error analysis
def create_error_analysis_plots(reconstruction_results):
    """Create comprehensive error analysis plots"""
    
    # Prepare data for plotting
    error_data = []
    
    for mesh_name, methods in reconstruction_results.items():
        for method_name, results in methods.items():
            error_data.append({
                'Mesh': mesh_name,
                'Method': method_name,
                'MSE': results['mse'],
                'MAE': results['mae'],
                'MSE_X': results['mse_per_axis'][0],
                'MSE_Y': results['mse_per_axis'][1],
                'MSE_Z': results['mse_per_axis'][2],
                'RelErr_X': results['relative_error'][0],
                'RelErr_Y': results['relative_error'][1],
                'RelErr_Z': results['relative_error'][2]
            })
    
    error_df = pd.DataFrame(error_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Overall MSE comparison
    meshes = error_df['Mesh'].unique()
    methods = error_df['Method'].unique()
    
    x = np.arange(len(meshes))
    width = 0.35
    
    mse_minmax = error_df[error_df['Method'] == 'minmax']['MSE'].values
    mse_unitsphere = error_df[error_df['Method'] == 'unitsphere']['MSE'].values
    
    axes[0,0].bar(x - width/2, mse_minmax, width, label='Min-Max', alpha=0.8)
    axes[0,0].bar(x + width/2, mse_unitsphere, width, label='Unit Sphere', alpha=0.8)
    axes[0,0].set_xlabel('Mesh')
    axes[0,0].set_ylabel('MSE')
    axes[0,0].set_title('Mean Squared Error by Method')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(meshes, rotation=45)
    axes[0,0].legend()
    
    # 2. Overall MAE comparison
    mae_minmax = error_df[error_df['Method'] == 'minmax']['MAE'].values
    mae_unitsphere = error_df[error_df['Method'] == 'unitsphere']['MAE'].values
    
    axes[0,1].bar(x - width/2, mae_minmax, width, label='Min-Max', alpha=0.8)
    axes[0,1].bar(x + width/2, mae_unitsphere, width, label='Unit Sphere', alpha=0.8)
    axes[0,1].set_xlabel('Mesh')
    axes[0,1].set_ylabel('MAE')
    axes[0,1].set_title('Mean Absolute Error by Method')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(meshes, rotation=45)
    axes[0,1].legend()
    
    # 3. Relative error per axis
    axes_labels = ['X', 'Y', 'Z']
    rel_errors_minmax = []
    rel_errors_unitsphere = []
    
    for axis in axes_labels:
        rel_errors_minmax.append(error_df[error_df['Method'] == 'minmax'][f'RelErr_{axis}'].mean())
        rel_errors_unitsphere.append(error_df[error_df['Method'] == 'unitsphere'][f'RelErr_{axis}'].mean())
    
    x_axis = np.arange(len(axes_labels))
    axes[1,0].bar(x_axis - width/2, rel_errors_minmax, width, label='Min-Max', alpha=0.8)
    axes[1,0].bar(x_axis + width/2, rel_errors_unitsphere, width, label='Unit Sphere', alpha=0.8)
    axes[1,0].set_xlabel('Axis')
    axes[1,0].set_ylabel('Relative Error (%)')
    axes[1,0].set_title('Average Relative Error per Axis')
    axes[1,0].set_xticks(x_axis)
    axes[1,0].set_xticklabels(axes_labels)
    axes[1,0].legend()
    
    # 4. Summary statistics table
    axes[1,1].axis('tight')
    axes[1,1].axis('off')
    
    summary_stats = error_df.groupby('Method').agg({
        'MSE': ['mean', 'std'],
        'MAE': ['mean', 'std']
    }).round(6)
    
    table_data = []
    for method in methods:
        method_data = error_df[error_df['Method'] == method]
        table_data.append([
            method,
            f"{method_data['MSE'].mean():.6f}",
            f"{method_data['MSE'].std():.6f}",
            f"{method_data['MAE'].mean():.6f}",
            f"{method_data['MAE'].std():.6f}"
        ])
    
    table = axes[1,1].table(cellText=table_data,
                           colLabels=['Method', 'MSE Mean', 'MSE Std', 'MAE Mean', 'MAE Std'],
                           cellLoc='center',
                           loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1,1].set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig('visualizations/error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Error analysis plots saved to visualizations/error_analysis.png")
    
    return error_df

# Create error analysis plots
error_df = create_error_analysis_plots(reconstruction_results)

# Generate comprehensive analysis report
def generate_analysis_report(error_df, reconstruction_results):
    """Generate comprehensive analysis report"""
    
    print("\\n" + "=" * 80)
    print("MESH NORMALIZATION, QUANTIZATION, AND ERROR ANALYSIS REPORT")
    print("=" * 80)
    
    # Overall comparison
    method_comparison = error_df.groupby('Method').agg({
        'MSE': ['mean', 'std', 'min', 'max'],
        'MAE': ['mean', 'std', 'min', 'max']
    }).round(8)
    
    print("\\n1. OVERALL METHOD COMPARISON:")
    print("-" * 40)
    print(method_comparison)
    
    # Best method determination
    avg_mse = error_df.groupby('Method')['MSE'].mean()
    best_method = avg_mse.idxmin()
    
    print(f"\\n2. BEST PERFORMING METHOD:")
    print("-" * 40)
    print(f"Method with lowest average MSE: {best_method}")
    print(f"Average MSE: {avg_mse[best_method]:.8f}")
    
    # Per-mesh analysis
    print(f"\\n3. PER-MESH PERFORMANCE:")
    print("-" * 40)
    
    for mesh in error_df['Mesh'].unique():
        mesh_data = error_df[error_df['Mesh'] == mesh]
        best_for_mesh = mesh_data.loc[mesh_data['MSE'].idxmin(), 'Method']
        best_mse = mesh_data['MSE'].min()
        print(f"{mesh}: Best method = {best_for_mesh} (MSE: {best_mse:.8f})")
    
    # Key observations and conclusions
    print(f"\\n4. KEY OBSERVATIONS AND CONCLUSIONS:")
    print("-" * 40)
    
    conclusions = []
    
    if avg_mse['minmax'] < avg_mse['unitsphere']:
        conclusions.append("• Min-Max normalization generally produces lower reconstruction errors.")
        conclusions.append("  This suggests that preserving the original aspect ratios is beneficial.")
    else:
        conclusions.append("• Unit Sphere normalization generally produces lower reconstruction errors.")
        conclusions.append("  This indicates that centering and uniform scaling is more robust.")
    
    # Check quantization effectiveness
    avg_rel_error = error_df[['RelErr_X', 'RelErr_Y', 'RelErr_Z']].mean().mean()
    
    if avg_rel_error < 1.0:
        conclusions.append(f"• Quantization with {n_bins} bins preserves mesh structure very well (<1% relative error).")
    elif avg_rel_error < 5.0:
        conclusions.append(f"• Quantization with {n_bins} bins provides acceptable quality (<5% relative error).")
    else:
        conclusions.append(f"• Quantization with {n_bins} bins may cause noticeable quality loss (>{avg_rel_error:.1f}% relative error).")
    
    conclusions.extend([
        "• The quantization process introduces systematic errors that depend on the normalization method.",
        "• Complex geometries may show different error patterns compared to simple shapes.",
        "• The choice of normalization method should consider the specific application requirements."
    ])
    
    for conclusion in conclusions:
        print(conclusion)
    
    print(f"\\n5. RECOMMENDATIONS:")
    print("-" * 40)
    print(f"• For this dataset, use {best_method} normalization for best accuracy.")
    print(f"• Consider increasing quantization bins if higher precision is needed.")
    print(f"• Validate results with additional mesh types and sizes.")
    
    return {
        'best_method': best_method,
        'avg_mse': avg_mse,
        'method_comparison': method_comparison,
        'conclusions': conclusions
    }

# Generate analysis report
analysis_report = generate_analysis_report(error_df, reconstruction_results)

print("\\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("Files generated:")
print("- Sample mesh files in data/")
print("- Processed mesh files in output/")
print("- Visualization plots in visualizations/")
print("- This complete analysis script")