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

### Citation

Please cite the paper if you use this code in your experiments:

```
@article{https://doi.org/10.1155/2023/4796536,
author = {Hesamipour, Sajjad and Balafar, Mohammad Ali and Mousazadeh, Saeed},
title = {Detecting Communities in Complex Networks Using an Adaptive Genetic Algorithm and Node Similarity-Based Encoding},
journal = {Complexity},
volume = {2023},
number = {1},
pages = {4796536},
doi = {https://doi.org/10.1155/2023/4796536},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1155/2023/4796536},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1155/2023/4796536},
abstract = {Detecting communities in complex networks can shed light on the essential characteristics and functions of the modeled phenomena. This topic has attracted researchers from both academia and industry. Among different community detection methods, genetic algorithms (GAs) have become popular. Considering the drawbacks of the currently used locus-based and solution-vector-based encodings to represent the individuals, in this article, we propose (1) a new node similarity-based encoding method, MST-based encoding, to represent a network partition as an individual, which can avoid the shortcomings of the previous encoding schemes. Then, we propose (2) a new adaptive genetic algorithm for the purpose of detecting communities in networks, along with (3) a new initial population generation function to improve the convergence time of the algorithm, and (4) a new sine-based adaptive mutation function which adjusts the mutations according to the improvement in the fitness value of the best individual in the population pool. The proposed method combines the similarity-based and modularity-optimization-based approaches to find communities in complex networks in an evolutionary framework. Besides the fact that the proposed encoding can avoid meaningless mutations or disconnected communities, we show that the new initial population generation function and the new adaptive mutation function can improve the convergence time of the algorithm. Several experiments and comparisons were conducted to verify the effectiveness of the proposed method using modularity and NMI measures on both real-world and synthetic datasets. The results show that the proposed method can find the communities in a significantly shorter time than other GAs while reaching a better trade-off in the different measures.},
year = {2023}
}
```
