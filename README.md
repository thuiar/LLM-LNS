# LLM-LNS

A framework that leverages LLMs to automatically generate and optimize heuristic algorithms for solving combinatorial optimization problems.

## Overview

This repository implements a dual-layer evolutionary framework that:
- Uses LLMs to generate heuristic algorithms automatically
- Employs evolutionary algorithms to optimize both prompts and generated algorithms
- Supports multiple classic combinatorial optimization problems

## Supported Problems

### MILP Problems
- **IS**: Independent Set Problem
- **MIKS**: Maximum Independent K-Set Problem  
- **MVC**: Minimum Vertex Cover Problem
- **SC**: Set Cover Problem

### Combinatorial Optimization Problems
- **Online Bin Packing**: Dynamic bin packing with online constraints
- **Traveling Salesman Problem**: Classic TSP variants

## Datasets

For MILP problems, datasets are avaliable [here](https://github.com/thuiar/MILPBench/tree/main/Benchmark%20Datasets).

For online BP problems, the dataset is involved in the code.

For TSP problems, datasets are available [here](https://github.com/mastqe/tsplib).

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Configure API Access

Replace `your_llm_endpoint` and `your_api_key` in the code with your actual LLM endpoint and API key.

### 2. Run Examples

```python
# Run Minimum Vertex Cover problem
python src/MILP\ Problems/MVC_eoh_change_prompt_ACP.py

# Run Independent Set problem  
python src/MILP\ Problems/IS_eoh_change_prompt_ACP.py

# Run Set Cover problem
python src/MILP\ Problems/SC_eoh_change_prompt_ACP.py
```

## Results

Experimental results and performance comparisons are available in [`results/results.md`](results/results.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.