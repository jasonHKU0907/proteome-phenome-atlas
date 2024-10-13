#!/bin/bash

pheno=$1

/public/home2/nodecw_group/UKB_trait/tmp/Proteomics_atlas/GWAS/gcta_1.94.0beta/gcta64 \
--mbfile /public/home2/nodecw_group/UKB_gene_v3_eur_38w_grm/plink_file.list \
--grm-sparse /public/home2/nodecw_group/UKB_gene_v3_eur_38w_grm/european \
--fastGWA-mlm-binary \
--pheno /public/home2/nodecw_group/UKB_trait/tmp/Proteomics_atlas/GWAS/data/Disease_supple/gwas_${pheno}.txt \
--qcovar /public/home2/nodecw_group/UKB_trait/Genetic/quan_cov.txt \
--covar /public/home2/nodecw_group/UKB_trait/Genetic/cate_cov.txt \
--threads 10 \
--maf 0.01 \
--geno 0.05 \
--keep /public/home2/nodecw_group/UKB_trait/tmp/Proteomics_atlas/GWAS/data/british_OLINK_mind_id.txt \
--out /public/home2/nodecw_group/tmp/dyt_GCTA/results_supple/geno_assoc_${pheno}
