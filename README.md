# Using Nature-Inspired Metaheuristics for Enhancing Melanoma Detection and Model Interpretability

## Table of Contents
- [Overview](#overview)
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Algorithms Implemented](#algorithms-implemented)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project explores the application of various nature-inspired metaheuristic algorithms for optimizing hyperparameters in Convolutional Neural Networks (CNNs) for melanoma skin cancer detection. Additionally, it investigates model interpretability using LIME (Local Interpretable Model-agnostic Explanations) enhanced with metaheuristic optimization techniques.

## Project Description

Melanoma is one of the most dangerous forms of skin cancer, and early detection is crucial for successful treatment. This research project focuses on:

1. **Hyperparameter Optimization**: Using nature-inspired metaheuristics to optimize CNN hyperparameters for melanoma classification
2. **Model Interpretability**: Enhancing LIME explanations using metaheuristic algorithms to improve understanding of model decisions

The project demonstrates how different optimization algorithms perform in finding optimal hyperparameters for a binary classification task (Benign vs. Malignant melanoma).

## Dataset

**Dataset**: [Melanoma Cancer Dataset](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset) by Bhavesh Mittal

### Dataset Statistics:
- **Training Set**: 7,128 images (or 9,504 images depending on split)
- **Validation Set**: 4,751 images (or 2,375 images depending on split)
- **Test Set**: 1,200 images (or 2,000 images depending on split)
- **Classes**: 
  - Benign
  - Malignant
- **Image Size**: 64×64 or 224×224 pixels (depending on notebook)
- **Format**: RGB images

### Data Preprocessing:
- Image normalization (rescaling to [0,1])
- Data augmentation:
  - Shear transformation (0.2)
  - Zoom (0.2)
  - Horizontal flip
- Train/validation split: 60/40 or 80/20 (depending on notebook)

## Methodology

### Phase 1: Hyperparameter Optimization

The project optimizes the following CNN hyperparameters using various metaheuristic algorithms:

- **filters1**: Number of filters in first convolutional layer [32, 64, 128]
- **filters2**: Number of filters in second convolutional layer [32, 64, 128]
- **dropout1**: Dropout rate for first dense layer [0.1, 0.3, 0.5, 0.7]
- **dropout2**: Dropout rate for second dense layer [0.1, 0.3, 0.5, 0.7]
- **learning_rate**: Learning rate for Adam optimizer [0.0001, 0.001, 0.01, 0.1]
- **activation**: Activation function ['relu', 'elu', 'gelu']
- **last_activation**: Output layer activation ['sigmoid']

### Phase 2: Model Interpretability Enhancement

Using LIME for model interpretability, enhanced with metaheuristic algorithms to optimize explanation quality by maximizing positive superpixel weights.

## Algorithms Implemented

### 1. **Firefly Algorithm (FF)**
- Inspired by the flashing behavior of fireflies
- Parameters: alpha (randomness), beta0 (attractiveness), gamma (absorption coefficient)
- Uses continuous optimization with attraction between fireflies

### 2. **Whale Optimization Algorithm (WOA)**
- Mimics the hunting behavior of humpback whales
- Parameters: b (spiral constant), population size, iterations
- Implements encircling prey, bubble-net attacking, and searching for prey behaviors

### 3. **Ant Colony Optimization (ACO)**
- Based on ant foraging behavior
- Parameters: number of ants, iterations, evaporation rate
- Uses pheromone trails to guide search

### 4. **Particle Swarm Optimization (PSO)**
- Inspired by bird flocking behavior
- Parameters: inertia weight (W), cognitive coefficient (C1), social coefficient (C2)
- Particles move towards personal best and global best positions

### 5. **Simulated Annealing (SA)**
- Based on metal annealing process
- Parameters: initial temperature, cooling rate, iterations per temperature
- Accepts worse solutions probabilistically to escape local optima

### 6. **Tabu Search (TL)**
- Uses memory to avoid revisiting solutions
- Parameters: tabu tenure, max iterations
- Implements aspiration criteria to override tabu status

### 7. **Grey Wolf Optimizer (GW)**
- Mimics the social hierarchy and hunting behavior of grey wolves
- Used for LIME optimization in Phase 2
- Parameters: population size, iterations

### 8. **Cuckoo Search (CS)**
- Based on cuckoo bird's brood parasitism
- Used for LIME optimization in Phase 2
- Parameters: number of nests, abandonment probability (pa), step size (alpha)

### 9. **Hill Climbing (HC)**
- Local search algorithm
- Used to optimize metaheuristic parameters (e.g., WOA's 'b' parameter, Firefly's 'alpha')
- Simple gradient-free optimization

## Model Architecture

The CNN architecture consists of:

```
Input Layer: (IMAGE_WIDTH, IMAGE_HEIGHT, 3)
    ↓
Conv2D(filters1, 3×3) + Activation
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(filters2, 3×3) + Activation
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(128, 3×3) + Activation
    ↓
MaxPooling2D(2×2)
    ↓
Flatten
    ↓
Dense(128) + Activation + Dropout(dropout1)
    ↓
Dense(64) + Activation + Dropout(dropout2)
    ↓
Dense(1) + Sigmoid
```

**Optimizer**: Adam  
**Loss Function**: Binary Crossentropy  
**Metrics**: Accuracy  
**Callbacks**: EarlyStopping, ReduceLROnPlateau

## Results

### Hyperparameter Optimization Results

| Algorithm | Test Accuracy | Validation Accuracy (Best Found) | Training Time (seconds) | Notes |
|-----------|---------------|----------------------------------|-------------------------|-------|
| **Base Model** | 75.55% | 78.49% | 543.04 | Baseline without optimization |
| **Firefly Algorithm (FF)** | 83.83% | ~82.69% | 3328 | Continuous optimization |
| **Whale Optimization (WOA)** | 80.75% | ~82.39% | 2185 | Spiral updating mechanism |
| **Ant Colony Optimization (ACO)** | 87.15% | ~79.98% | 26,709.78 | Pheromone-based search |
| **Particle Swarm Optimization (PSO)** | 87.10% | ~81.25% | 15,243.95 | Swarm intelligence |
| **Simulated Annealing (SA)** | 88.95% | ~80.03% | 29,169.21 | Temperature-based acceptance |
| **Tabu Search (TL)** | 90.10% | ~82.18% | 36,017.79 | Memory-based search |
| **WOA + Hill Climbing** | 81.67% | ~81.88% | - | Hybrid approach |
| **Firefly + Hill Climbing** | 80.92% | ~82.64% | - | Hybrid approach |

### Model Interpretability Results (Phase 2)

| Algorithm | LIME Weight Improvement | Application |
|-----------|------------------------|--------------|
| **Simulated Annealing** | Initial: 0.0214 → Final: 0.0391 | LIME explanation optimization |
| **Tabu Search** | Initial: 0.0217 → Final: 0.0445 | LIME explanation optimization |
| **Grey Wolf Optimizer** | Initial: ~0.02 → Final: 0.0492 | LIME explanation optimization |
| **Cuckoo Search** | Initial: ~0.02 → Final: 0.0369 | LIME explanation optimization |

### Key Findings

1. **Best Performance**: Tabu Search achieved the highest test accuracy of **90.10%**
2. **Efficiency**: Firefly and WOA algorithms showed good balance between performance and computational cost
3. **Hybrid Approaches**: Combining metaheuristics with Hill Climbing showed promising results
4. **Interpretability**: Metaheuristic-enhanced LIME explanations improved superpixel weight scores significantly

## Project Structure

```
Notebooks/
├── nic_project_FF.ipynb                          # Firefly Algorithm implementation
├── nic_project_WOA.ipynb                          # Whale Optimization Algorithm
├── nic-project-ACO.ipynb                          # Ant Colony Optimization
├── nic-project-PSO.ipynb                          # Particle Swarm Optimization
├── nic-project-SA.ipynb                           # Simulated Annealing
├── nic-project-TL.ipynb                           # Tabu Search
├── nic-project-phase-2-LIME-with-GW-&-CS-&-TL-&-SA.ipynb  # LIME interpretability with metaheuristics
└── nic-project-phase-2-WOA-&-FF-with-Hill-Climbing.ipynb  # Hybrid approaches
```

## Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or Google Colab
- GPU recommended (for faster training)

### Install Dependencies

```bash
pip install tensorflow keras numpy matplotlib scikit-learn scikit-image lime kagglehub
```

Or using conda:

```bash
conda install tensorflow keras numpy matplotlib scikit-learn scikit-image
pip install lime kagglehub
```

## Usage

### Running Hyperparameter Optimization

1. **Download the dataset** (automatically handled in notebooks using `kagglehub`):
   ```python
   import kagglehub
   path = kagglehub.dataset_download("bhaveshmittal/melanoma-cancer-dataset")
   ```

2. **Run any optimization notebook**:
   - Open the desired notebook (e.g., `nic-project-SA.ipynb`)
   - Execute all cells sequentially
   - The notebook will:
     - Download and preprocess data
     - Run the optimization algorithm
     - Train the final optimized model
     - Evaluate on test set
     - Save the model

### Example: Running Simulated Annealing

```python
# Set search space
search_space = {
    'filters1': [32, 128, 256],
    'filters2': [32, 128, 256],
    'dropout1': [0.1, 0.3, 0.5, 0.7],
    'dropout2': [0.1, 0.3, 0.5, 0.7],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'activation': ['relu', 'elu', 'gelu'],
    'last_activation': ['sigmoid']
}

# Run optimization
best_state, best_acc = simulated_annealing_search()

# Train final model
optimized_model = build_and_train_model(best_state)
```

### Running LIME Interpretability Enhancement

1. Load a trained model
2. Run the Phase 2 notebook: `nic-project-phase-2-LIME-with-GW-&-CS-&-TL-&-SA.ipynb`
3. Select an image for explanation
4. Run metaheuristic-enhanced LIME optimization

## Dependencies

### Core Libraries
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **scikit-learn**: Machine learning utilities
- **scikit-image**: Image processing

### Specialized Libraries
- **LIME**: Local Interpretable Model-agnostic Explanations
- **kagglehub**: Kaggle dataset download

### Version Recommendations
```
tensorflow >= 2.10.0
keras >= 2.10.0
numpy >= 1.21.0
matplotlib >= 3.5.0
scikit-learn >= 1.0.0
scikit-image >= 0.19.0
lime >= 0.2.0
kagglehub >= 0.2.0
```

## Configuration

### Key Parameters

**Image Processing**:
- `IMAGE_WIDTH`: 64 or 224 pixels
- `IMAGE_HEIGHT`: 64 or 224 pixels
- `BATCH_SIZE`: 32

**Optimization**:
- Population size: 3-5 individuals
- Max iterations: 3-10 (varies by algorithm)
- Evaluation epochs: 4 epochs per candidate
- Final training epochs: 20 epochs

**Model Training**:
- Optimizer: Adam
- Loss: Binary Crossentropy
- Early stopping patience: 3 epochs
- Learning rate reduction factor: 0.2

## Notes

- All notebooks use random seed 42 for reproducibility
- Models are saved in `.keras` format
- Training times vary significantly based on:
  - Algorithm complexity
  - Number of iterations
  - Hardware (CPU vs GPU)
- Some notebooks use different image sizes (64×64 vs 224×224) for computational efficiency

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is for educational and research purposes. Please ensure you comply with the Kaggle dataset license when using the melanoma dataset.

## Acknowledgments

- Dataset: [Melanoma Cancer Dataset](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset) by Bhavesh Mittal
- LIME: [LIME - Local Interpretable Model-agnostic Explanations](https://github.com/marcotcr/lime)
- TensorFlow/Keras team for the deep learning framework

## Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This project demonstrates the application of nature-inspired metaheuristics in deep learning hyperparameter optimization and model interpretability. Results may vary based on hardware, random seeds, and dataset versions.

