# Hyperparameter Context Engineering Tour

A research project exploring the effectiveness of Large Language Models (LLMs) in hyperparameter optimization through in-context learning and sampling strategies. This project compares LLM-based optimization approaches against traditional uniform sampling across various benchmark functions.

## 🎯 Project Overview

This project investigates how LLMs can be leveraged for hyperparameter optimization by:
- **In-Context Learning (ICL)**: Using LLMs to learn from example solution-score pairs and generate improved candidates
- **Direct Sampling**: Comparing LLM-generated samples against uniform random sampling
- **Prior Knowledge Integration**: Testing the impact of function-specific characteristics on optimization performance


## 🧪 Supported Benchmark Functions

The project includes implementations of classic optimization benchmark functions:

- **Rastrigin**: Multi-modal function with many local minima
- **Rosenbrock**: Valley function with narrow, curved minimum
- **Griewank**: Function with many local minima on a parabolic surface
- **Michalewicz**: Highly multi-modal with deceptive local optima
- **Schwefel**: Function with global optimum at boundary
- **Sphere**: Simple convex function with single global minimum

## 🚀 Quick Start

### 1. Environment Setup

```bash
cp env.sample .env
```

### 2. Configure Environment Variables

Edit the `.env` file with your OpenAI credentials:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1
```

To test locally with an Ollama endpoint, use following settings:
```
OPENAI_API_KEY=no_key_required
OPENAI_MODEL=gemma3n:latest
OPENAI_BASE_URL=http://localhost:11434/v1
```

### 3. Run Experiments

#### Sampling Experiment
```bash
python3 exp_sampling.py --help
python3 exp_sampling.py
```

#### In-Context Learning Experiment
```bash
python3 exp_icl.py --help
python3 exp_icl.py
```

#### Experiment Results
You can find the experiment results in the CSV files located in the `exp_results/` directory.


### 4. Utility

#### Generate Function Visualizations
```bash
python3 utils/plot_all.py
```
This utility generates 3D visualizations of all benchmark functions and saves them in the `figs/` directory.
