# Meta-Learning and Targeted Differential Privacy to Improve the Accuracy–Privacy Trade–off in Recommendations

Submitted to the Late-Breaking-Results track at UMAP'26.

## Abstract
Balancing differential privacy (DP) with recommendation accuracy remains a key challenge in privacy-preserving recommender systems, as noise introduced to achieve DP can significantly degrade recommendation performance. 
We propose a two-stage approach addressing this trade-off at both the data and model levels. 
At the data level, we selectively apply DP to the most stereotypical or sensitive parts of user data, reducing unnecessary perturbation, i.e., targeted DP. 
At the model level, we leverage meta-learning to improve robustness to residual DP-noise and enable fine-grained adaptation to user-specific patterns.
Our results show that targeted DP achieves higher recommendation accuracy than uniformly applying DP across the entire dataset, while meta-learning further improves performance over standard recommender models.
We also observe lower empirical privacy risk compared to uniformly applied DP and full DP baselines.
Overall, a targeted application of DP combined with meta-learning enables more accurate and private recommender systems.

## Instructions
* <i>Dataset Preprocessing, Splitting, and application of DP</i>: Applies core-pruning (BX), splits datasets into training-, validation-, and testsets. Applies DP to the dataset depending on the value of the data budget $\beta$ (entire dataset, random subset, targeted application).
```
data/prepare_dataset.py
```
* <i>Hyperparameter Tuning</i>: Tune hyperparameters (learning rate, regularization factor) of MetaMF and NoMetaMF for both datasets and all values of the privacy budget $\epsilon$.
```
hyperparameter_tuning_meta.py
```
```
hyperparameter_tuning_nometa.py
```
Then, the best hyperparameters (lowest MAE on the validation-set) can be selected via
```
hypertuning_results/hypertuning_results.ipynb
```

* <i>run_recommender.py</i> runs the recommendation model on a given dataset
* <i>attacker.py</i> runs the neural attacker on a given dataste



## Requirements


## Contributors
* Peter Müllner, Know-Center GmbH, Graz, pmuellner [AT] know [MINUS] center [DOT] at (Contact)
* Markus Schedl, JKU and LIT, Linz
* Dominik Kowald, Know-Center GmbH and Graz University, Graz
* Elisabeth Lex, Graz University of Technology, Graz
