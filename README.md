## Genetic Algorithm for Community Discovery

Reference implementation of the method presented in [“A novel evolutionary approach for overlapping community detection in complex networks”](https://onlinelibrary.wiley.com/doi/10.1155/2023/4796536).

### Repository Layout
- `src/community_ga/main.py` – experiment runner and helper utilities.
- `src/community_ga/chromosom.py` – chromosome representation and GA operators.
- `scripts/statistical_tests.py` – Friedman/Autorank tests for the paper’s tables.

### Environment
1. Create a virtual environment (Python 3.8+ recommended).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Data
The pipeline expects adjacency matrices in Matrix Market (`*.mtx`) form. Place the files next to `main.py` (or update the loader to point to a dedicated `data/` folder). Datasets used in the paper (Karate, Polbooks, LFR, etc.) are publicly available from their original sources.

### Running Experiments
Launch the interactive driver:
```
python -m community_ga.main
```
The script lists all `.mtx` files in the working directory, then prompts for the dataset index, population size, modularity threshold, mutation probability, sine stepping parameter, experiment count, convergence limit, and mutation-function id. Generated community assignments land in `<graph>.out` plus auxiliary `.png/.maxs.out` files.

Key parameters are defined near the bottom of `main.py`. Adjust prompts or set sensible defaults before publication if you want a non-interactive workflow.

### Statistical Tests
To reproduce the statistical analysis figures/tables:
```
python scripts/statistical_tests.py
```
The script logs Friedman/Autorank summaries and emits plots mirroring the paper.

### Utilities
- `scripts/compute_q.py` recomputes the modularity (Q) of a saved `{dataset}.out` file:
  ```
  python scripts/compute_q.py --graph data/karate.mtx --communities data/karate.out
  ```
  Add `--weighted` if the graph file includes edge weights.

### Releasing the Code
- Keep generated `.out/.png` artifacts out of version control (they can be regenerated).
- Document any dataset preprocessing steps in this README if you customize them further.
- Cite the paper whenever this implementation is used in academic work.

