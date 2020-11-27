# Setup:

```console
pip install requirements.txt
```

# Breast Canser Usage Example:

```console
python bfs_dr.py --chromosomes 200 --features 30 \
				 --generation 3 --parents 10 --selection-method roulette_wheel \
				 --crossover-method multi_point --mutation-method flipping --mutation-rate 0.20
```