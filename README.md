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
* <i>Hyperparameter Tuning</i>: Tune hyperparameters (learning rate, regularization factor) of MetaMF[1] and NoMetaMF for both datasets and all values of the privacy budget $\epsilon$.
```
hyperparameter_tuning_meta.py
```
```
hyperparameter_tuning_nometa.py
```
Then, the best hyperparameters (lowest MAE on the validation-set) can be selected and saved via
```
hypertuning_results/hypertuning_results.ipynb
```
* <i>Run Recommendation Experiments</i>: Perform all recommendation experiments present in the paper, i.e., use MetaMF and NoMetaMF for both datasets, different $\epsilon$-values, different $\beta$-values and random or targeted application of DP. For example, 
```
run_recommender.py --model MetaMF --dataset ml1m --random_dp --seed "0 1 2"
```
The results can be analyzed via
```
results/analysis_meta_nometa.py
```
* <i>Run Attacker</i>: Run the neural attacker to quantify the emprical privacy risk. This can be done for both datasets, different $\epsilon$-values, different $\beta$-values and random or targeted application of DP. ALso, no hypertuning can be performed, for every run, or only for the first run. For example,
```
attacker.py --dataset ml1m --method random_dp --hypertuning first
```
The results can be analyzed via
```
results/analysis_attacker.py
```
* <i>Analyze the Trade-Off</i>: The final trade-off between recommendation accuracy and empirical privacy risk can be plotted via
```
results/analysis_tradeoff.py
```

## Contributors
* Peter Müllner, Know-Center GmbH, Graz, pmuellner [AT] know [MINUS] center [DOT] at (Contact)
* Markus Schedl, JKU and LIT, Linz
* Dominik Kowald, Know-Center GmbH and Graz University, Graz
* Elisabeth Lex, Graz University of Technology, Graz

[1] Lin, Yujie, et al. "Meta matrix factorization for federated rating predictions." Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval. 2020.
