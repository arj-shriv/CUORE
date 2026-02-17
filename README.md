# NeuralODE for Time-Series Signal Modeling  

## Overview  
This project implements a **Neural Ordinary Differential Equation (Neural ODE)** architecture in JAX and Equinox to model the continuous-time dynamics of real-world signals. Neural ODEs treat hidden state evolution as a differential equation parameterized by a neural network, allowing the model to learn realistic time-based behavior rather than relying on simple regression.  

## Key Contributions  
- **Generative & Differentiable Signal Model**: Built a model capable of simulating realistic signals for downstream tasks such as classification and optimization.  
- **Vectorized Training**: Scaled training from single-signal inputs to fully batched, vectorized training on 20+ signals at once, drastically improving throughput.  
- **Performance**: Achieved **5× faster training** and ~20% better scalability compared to earlier methods, demonstrating efficiency and reproducibility.  
- **Interpretability**: Differentiable modeling provides insights into how input variables evolve over time and supports gradient-based optimization.  

## Project Notebooks  
- **Linear_NODE.ipynb** – Simple example applying a Neural ODE to sine waves.  
- **ReadingSignals.ipynb** – Training on a single signal.  
- **MultipleSignals.ipynb** – Sequential training on 3 signals, one at a time.  
- **Vectorization.ipynb** – Vectorized training on 3 signals woven together.  
- **vectorization20.ipynb** – **Main notebook**: scalable vectorized training on 20+ signals simultaneously, demonstrating the final optimized workflow.  

## Why It Matters  
- Provides a data-driven approach to modeling time-series signals.  
- Enables faster and more flexible **signal simulation** for experimental or applied analysis.  
- Offers a generalizable method for continuous-time modeling applicable to IoT, finance, healthcare, and other domains.  
