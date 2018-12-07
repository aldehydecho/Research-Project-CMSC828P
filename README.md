# Research Project - Extracting signatures using f-LDA

Source code for research project about extracting signatures using factorial LDA (f-LDA) (Paul, et al. [2]), on dataset from Alexandrov, et al. [2]. In brief, the experiments are:

1. Perplexity comparison between 2-factor and 1-factor baseline f-LDA.
2. Mutation signature extraction using f-LDA.
3. Smoking history prediction using results of f-LDA.

Additional details of the experiments, data, and conclusions can be found in the [`report/`](report/).

## Setup

The source code is written in Python 3. We use `snakemake` to manage the workflow. We suggest using Conda to install dependencies, which you can do directly from the provided [`environment.yml`](environment.yml) file.

    conda env create -f environment.yml
    source activate reproducing-nikzainal2016-env

## Usage

To generate all the figures in the report and related statistical data, simply run:

    snakemake all

The data like accuracy, perplexity on testing data will be printed out on screen during the execution of the scripts.

This will take roughly half a day on a modern CPU to run one f-LDA training script. For example, runing f-LDA on whole dataset from Alexandrov, et al. [2]:

    snakemake run_flda

We also provide pre-trained models in [Google Drive](https://drive.google.com/drive/folders/1BqihHQwHQaiw_zZ3m0x_zhr5wzdPV_BR?usp=sharing). You can download them and put in the [`output/`](output/) folder, and then run "snakemake all" to get final results.



### Configuration

Additional configuration options are detailed at the beginning of the [`Snakefile`](Snakefile).

## References
1. Paul, et al. (2012) "Factorial LDA: Sparse Multi-Dimensional Text Models" _NIPS_
2. Alexandrov, et al. (2016) "Mutational signatures associated with tobacco smoking in human cancer." _Science_ 354(6312), pages 618-622. [doi: 10.1126/science.aag0299](https://doi.org/10.1126/science.aag0299)
