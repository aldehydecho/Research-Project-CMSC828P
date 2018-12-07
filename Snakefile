################################################################################
# SETUP
################################################################################
# Modules
from os.path import join


# Directories
DATA_DIR = 'data'
SRC_DIR = 'src'
OUTPUT_DIR = 'output' 
REPORT_DIR = 'report'
FIGS_OUTPUT_DIR = join(REPORT_DIR, 'figs')


# Files
SBS96_COUNTS = join(DATA_DIR, 'counts.Alexandrov2016.SBS-96.tsv')
CLINICAL = join(DATA_DIR, 'clinical.Alexandrov2016.tsv')
SAMPLES = join(DATA_DIR, 'samples.ALexandrov2016.tsv')
SUPP = join(DATA_DIR, 'aag0299_Tables_S1_to_S6.xlsx')

FLDA_OUTPUT = join(OUTPUT_DIR, 'storage_400.pkl')
FLDA_PERPLEXITY_OUTPUT = join(OUTPUT_DIR, 'storage_perplexity_250.pkl')
FLDA_PERPLEXITY_TEST_OUTPUT = join(OUTPUT_DIR, 'storage_perplexity_final.pkl')
FLDA_BASELINE_OUTPUT = join(OUTPUT_DIR, 'storage_baseline_400.pkl')
FLDA_PERPLEXITY_BASELINE_OUTPUT = join(OUTPUT_DIR, 'storage_perplexity_baseline_250.pkl')
FLDA_PERPLEXITY_BASELINE_TEST_OUTPUT = join(OUTPUT_DIR, 'storage_perplexity_baseline_final.pkl')

# Scripts
FLDA_PY = join(SRC_DIR, 'flda.py')
FLDA_PERPLEXITY_PY = join(SRC_DIR, 'flda_perplexity.py')
FLDA_PERPLEXITY_TEST_PY = join(SRC_DIR, 'flda_perplexity_test.py')
FLDA_BASELINE_PY = join(SRC_DIR, 'flda_baseline.py')
FLDA_PERPLEXITY_BASELINE_PY = join(SRC_DIR, 'flda_perplexity_baseline.py')
FLDA_PERPLEXITY_BASELINE_TEST_PY = join(SRC_DIR, 'flda_perplexity_baseline_test.py')
ANALYSIS_PY = join(SRC_DIR, 'Analysis.py')

# Name
TABLE_NAME = 'Table\ S6'

# Output figures
F1 = join(FIGS_OUTPUT_DIR, 'fLDA_vs_original.png')
F2 = join(FIGS_OUTPUT_DIR, 'fLDA_vs_self.png')
F3 = join(FIGS_OUTPUT_DIR, 'fLDA_baseline_vs_original.png')
F4 = join(FIGS_OUTPUT_DIR, 'fLDA_baseline_vs_self.png')


################################################################################
# GENRAL RULES
################################################################################
rule all:
    input:
        F1,
        F2,
        F3,
        F4,
        FLDA_PERPLEXITY_TEST_OUTPUT,
        FLDA_PERPLEXITY_BASELINE_TEST_OUTPUT

rule run_flda:
    input:
        SBS96_COUNTS
    output:
        FLDA_OUTPUT
    shell:
        'python {FLDA_PY} --input_file_sbs {input}'

rule run_flda_perplexity:
    input:
        SBS96_COUNTS
    output:
        FLDA_PERPLEXITY_OUTPUT
    shell:
        'python {FLDA_PERPLEXITY_PY} --input_file_sbs {input}'

rule run_flda_perplexity_test:
    input:
        i1 = SBS96_COUNTS,
        i2 = FLDA_PERPLEXITY_OUTPUT
    output:
        FLDA_PERPLEXITY_TEST_OUTPUT
    shell:
        'python {FLDA_PERPLEXITY_TEST_PY} --input_file_sbs {input.i1} --input_model {input.i2} --output_model {output}'

rule run_flda_baseline:
    input:
        SBS96_COUNTS
    output:
        FLDA_BASELINE_OUTPUT
    shell:
        'python {FLDA_BASELINE_PY} --input_file_sbs {input}'

rule run_flda_perplexity_baseline:
    input:
        SBS96_COUNTS
    output:
        FLDA_PERPLEXITY_BASELINE_OUTPUT
    shell:
        'python {FLDA_PERPLEXITY_BASELINE_PY} --input_file_sbs {input}'

rule run_flda_perplexity_baseline_test:
    input:
        i1 = SBS96_COUNTS,
        i2 = FLDA_PERPLEXITY_BASELINE_OUTPUT
    output:
        FLDA_PERPLEXITY_BASELINE_TEST_OUTPUT
    shell:
        'python {FLDA_PERPLEXITY_BASELINE_TEST_PY} --input_file_sbs {input.i1} --input_model {input.i2} --output_model {output}'

rule analysis:
    input:
        i1 = FLDA_OUTPUT,
        i2 = FLDA_BASELINE_OUTPUT,
        i3 = SBS96_COUNTS,
        i4 = CLINICAL,
        i5 = SAMPLES,
        i6 = SUPP
    params:
        TABLE_NAME
    output:
        o1 = F1,
        o2 = F2,
        o3 = F3,
        o4 = F4

    shell:
        'python {ANALYSIS_PY} --input_file_sbs {input.i3} --input_file_sample {input.i5} --input_file_clinical {input.i4} --input_file_flda_result {input.i1} --input_file_flda_baseline_result {input.i2} --input_file_supp {input.i6} --input_file_supp_table_name {params} --output_fig_1 {output.o1} --output_fig_2 {output.o2} --output_fig_3 {output.o3} --output_fig_4 {output.o4}'


