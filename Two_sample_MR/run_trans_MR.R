#### MR:trans pQTL → diseases ####
library(TwoSampleMR)
library(MRInstruments)
library(data.table)
library(R.utils)
library(tidyverse)
library(openxlsx)

type <- 'Incident'
data <- read.xlsx(paste0("/public/home/dengyueting/Proteomics/Atlas/associations/results/result_merge_",type,"_All.xlsx"),sep=',')
a <- as.data.frame(table(data$outcome))
i=1
for (i in 1:nrow(a)){
  print(i)
  disease <- as.character(a[i,1])
  dt <- subset(data,data$outcome == disease)
  pro <- as.data.frame(table(dt$Pro_code))
  pro <- pro[,1]
  IV <- fread("/public/home/dengyueting/Proteomics/Atlas/MR/QTL/data/proteinIV_UKB_trans.txt") %>% 
    filter(Protein %in% pro)
  IV <- as.data.frame(IV)
  if (nrow(IV) == 0){
    
  } else {
    IV <- IV[order(IV$P),]
    ## TwoSampleMR ##
    ins <- format_data(IV,type = "exposure",header = T,phenotype_col = "Protein",snp_col = "rsid",beta_col = "BETA",se_col = "SE",eaf_col = "A1FREQ",effect_allele_col = "ALLELE1",other_allele_col = "ALLELE0",pval_col = "P")
    tryCatch({
      pheno <- as.data.frame(fread(paste0("/public/home3/FinnGen-gwas/finngen_R9_",disease,".gz"),header = T))
      if (nrow(pheno) > 0){
        pheno$pheno <- disease
        out <- format_data(pheno,type = "outcome",header = T,snps = ins$SNP,phenotype_col = "pheno",snp_col = "rsids",beta_col = "beta",se_col = "sebeta",effect_allele_col ="alt",other_allele_col = "ref",eaf_col = "af_alt",pval_col = "pval")
        harmo <- harmonise_data(exposure_dat = ins,outcome_dat = out)
        mr_res <- mr(harmo,method_list = c("mr_wald_ratio","mr_ivw"))
        results_OR <- generate_odds_ratios(mr_res)
        results_OR <- results_OR[,-c(1,2)]
        write.table(results_OR,paste0("/public/home/dengyueting/Proteomics/Atlas/MR/transMR/results/",type,"/",disease,"_trans_MR.csv"),col.names = T,row.names = F,quote = F,sep = ",")
      } else {
        pheno <- as.data.frame(fread(paste0("/public/home2/nodecw_group/tmp/dyt_GCTA/results/geno_assoc_",disease,".fastGWA"),header = T))
        if (nrow(pheno) > 0){
          pheno$pheno <- disease
          out <- format_data(pheno,type = "outcome",header = T,snps = ins$SNP,phenotype_col = "pheno",snp_col = "SNP",beta_col = "BETA",se_col = "SE",effect_allele_col ="A1",other_allele_col = "A2",eaf_col = "AF1",pval_col = "P")
          harmo <- harmonise_data(exposure_dat = ins,outcome_dat = out)
          mr_res <- mr(harmo,method_list = c("mr_wald_ratio","mr_ivw"))
          results_OR <- generate_odds_ratios(mr_res)
          results_OR <- results_OR[,-c(1,2)]
          write.table(results_OR,paste0("/public/home/dengyueting/Proteomics/Atlas/MR/transMR/results/",type,"/",disease,"_trans_MR.csv"),col.names = T,row.names = F,quote = F,sep = ",")
        }
      }
    }, error = function(e){
      print(paste0('error_transMR:',disease))
    })
  }
}


