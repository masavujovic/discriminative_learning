# Discriminative Learning 

This repo contains the code base for a series of discriminative learning models (implementing a simplified version of the Rescorla-Wagner learning rule) as reported in my PhD [thesis](https://discovery.ucl.ac.uk/id/eprint/10111567/1/Vujovic_PhD_final_20201005.pdf):

* Vujović, M. (2020). Exploring language learning as uncertainty reduction using artificial language learning. University College London. 

as well as here:

* Vujović, M., Ramscar, M., & Wonnacott, E. (2021). Laguage learning as uncertainty reduction: The role of prediction error in linguistic generalization and item learning. <em>Journal of Memory and Language</em>. [[pre-print](https://osf.io/f2n9d/)]

To replicate the results from the paper, run the following (where <code>beta</code> is the learning rate and <code>n_trials</code> is the number of trials to run the model for; in Vujović et al., we used <code>0.01</code> and <code>7000</code>, respectively).

```python
from experiment_3 import Results

results = Results(beta, n_trials)
results.plot_paper()
```
