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

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Configure API Access

Set up your LLM API credentials (e.g., OpenAI):
```bash
export OPENAI_API_KEY="your_api_key_here"
```

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