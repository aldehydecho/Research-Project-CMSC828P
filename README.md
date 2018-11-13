# Research Project

## Proposal

### Experiment
Use Factorial LDA [1] to analysis datasets in Alexandrov, et al. [2], assuming differet tissue types and non-smokers vs smokers as two different factors. This can help us understand smoking's influence over mutation signatures of different tissues, better than post-analysis of directly running NMF on the whole dataset or running NMF individually on different tissues.

### Data and resources
1. We can use data for "Reproducing Alexandrov, et al. " projects.
2. We can modify [code from f-LDA author](http://cmci.colorado.edu/~mpaul/downloads/flda.php). The original code is written in Java, we can reproduce it to Python.

### Validation plans
After get topics for different tissues and smoker vs non-smokers groups, we can check the influence of smoking to different tissues' cancers. We could then compare the results with the ones in Alexandrov, et al. [2], to see whether the conclusions are consistent.


## References
1. Paul, et al. (2012) "Factorial LDA: Sparse Multi-Dimensional Text Models" _NIPS_
2. Alexandrov, et al. (2016) "Mutational signatures associated with tobacco smoking in human cancer." _Science_ 354(6312), pages 618-622. [doi: 10.1126/science.aag0299](https://doi.org/10.1126/science.aag0299)