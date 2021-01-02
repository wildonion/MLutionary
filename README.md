## ⚙️ Setup

```console
pip install requirements.txt
```

## 💻 Breast Canser Usage Example

```console
python bfs_dr.py --chromosomes 200 --features 30 \
				 --generation 3 --parents 10 --selection-method rank \
				 --crossover-method multi_point --mutation-method flipping --mutation-rate 0.20
```

## 📊 Plot of Breast Cancer Dataset Fitness Generation Process

<p align="center">
    <img src="https://github.com/wildonion/MLutionary-Evolver/blob/master/fitness_generation.png">
</p>

## 📌 TODOs

* Add Selection Methods
* Add Crossover Methods
* Add Mutation Methods
* Add Replacement Methods
* Test The GA Model on other Datasets' Features
