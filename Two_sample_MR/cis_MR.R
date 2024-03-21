####IV selection####
Criteria:
1) Selected SNPs at P < 5e-8
2) LD clumping (r2 < 0.01 within 1Mb)
3) Excluded MHC region (chr6: 25.5-34.0Mb)
4) Selected SNPs with a F-statistic > 10
5) Excluded SNPs associated with more than five proteins (highly pleiotropic)

#### 1&2.LD clumping ####
/public/home/dengyueting/Proteomics/Atlas/MR/COLOC/code/cis_IV.sh
# 检查日志 #
cd /public/home/dengyueting/Proteomics/Atlas/MR/COLOC/code/nohup/
for i in {1..11}
do
echo ${i} >> 0202_cisIV_check.txt
grep 'Error' 0202_cisIV_${i}.log >> 0202_cisIV_check.txt
done

#### 3.Excluded MHC region ####
rm(list=ls())
library(utils)
array <- list.files("/public/home/dengyueting/Proteomics/Atlas/MR/COLOC/data/cisSNP_r2_0.01_clumped/",pattern = "clumps$")
i=1
IV_all <- NULL
pb <- txtProgressBar(style=3)
star_time <- Sys.time()
### 先跑一个i=1的给一个非空的n值 ###
for (i in 2:length(array)){
  n <- nrow(IV_all)
  name <- array[i]
  cisfilename <- gsub("clumped_","",name)
  cisfilename <- gsub(".clumps","",cisfilename)
  cisfilename <- gsub("chr23","chrX",cisfilename)
  pro <- fread(paste0("/public/home/dengyueting/Proteomics/Atlas/MR/COLOC/data/cisSNP/",cisfilename,".txt"))
  dt <- fread(paste0("/public/home/dengyueting/Proteomics/Atlas/MR/COLOC/data/cisSNP_r2_0.01_clumped/",array[i]))
  dt <- dt[,c(3,4)]
  IV <- merge(dt,pro,by.x = 'ID',by.y = 'rsid',all.x = T)
  # 3.Excluded MHC region #
  if (IV$chr[1] == 6){
    idx <- which(IV$pos >= 25500000&IV$pos <= 34000000)
    if (length(idx) == 0){
    } else {
      IV <- IV[-idx,]
    }
  }
  pro <- str_split(cisfilename,"_")[[1]][3]
  pro <- str_split(pro,":")[[1]][1]
  IV$Pro_code <- pro
  IV_all <- as.data.frame(rbind(IV_all,IV))
  n2<-nrow(IV_all)
  # 目的：检查是不是每个clump文件都成功合并到IV_all中 #
  if (n2>n){} else {
    print(paste0(i,":",pro))
  }
  setTxtProgressBar(pb, i/length(array))
}
end_time <- Sys.time()
close(pb)
run_time <- end_time - star_time #2min
t <- as.data.frame(table(IV_all$Pro_code))

#### 4.Selected SNPs with a F-statistic > 10 ####
IV_all$R2 <- 2*IV_all$MAF*(1-IV_all$MAF)*(IV_all$BETA^2)
IV_all$F_sta <- IV_all$R2*(IV_all$N-2)/(1-IV_all$R2)
IV_all <- IV_all[which(IV_all$F_sta > 10),]
# 似乎没有SNP被过滤

#### 5.Excluded SNPs associated with more than five proteins (highly pleiotropic) ####
rsid_count <- aggregate(IV_all$ID, by = list(IV_all$ID), FUN = length)
colnames(rsid_count)[1] <- "rsid"
colnames(rsid_count)[2] <- "N"
rsid_count <- rsid_count[which(rsid_count$N <= 5),]
IV <- as.data.frame(rsid_count[,1])
colnames(IV)[1] <- "rsid"
IV$keep <- 1
IV <- merge(IV,IV_all,by.x = "rsid",by.y = "ID",all.y = T)
IV <- subset(IV,is.na(IV$keep) == F)
as.data.frame(colnames(IV))
IV <- IV[,c(27,1,6:12,21,14,16,17,19,3,28,29)]

#从何宇原本数据中提取遗漏的DCTD, DDC和DFFA cis pQTL
library(plyr)
IV_hy <- fread("/public/home/dengyueting/Proteomics/Atlas/MR/QTL/data/IV.csv")
iv <- subset(IV_hy,IV_hy$PROTEIN == 'DCTD' | IV_hy$PROTEIN == 'DDC' | IV_hy$PROTEIN == 'DFFA')
iv <- iv[c(8,1:4,9,11:13,17,21:23,7,10),]
colnames(iv)[1] <- 'Pro_code'
colnames(iv)[2] <- 'rsid'
colnames(iv)[4] <- 'chr'
colnames(iv)[6] <- 'pos'
iv$MAF <- iv$A1FREQ
colnames(iv)[16] <- 'F_sta'
IV <- rbind.fill(IV,iv)
IV <- IV[,-c(18,19)]
write.table(IV,"/public/home/dengyueting/Proteomics/Atlas/MR/cis_MR/data/cispQTL_IV_UKB.txt",row.names = F,col.names = T,sep = "\t",quote = F)

N_pro <- as.data.frame(table(IV$Pro_code))
#### MR analysis ####
Pipeline:
1. cis pQTL → Disease Finngen有的用Finngen, 没有用UKB
2. trans pQTL → Disease Finngen有的用Finngen, 没有用UKB
3. Disease → cis pQTL
4. Disease → trans pQTL

library(openxlsx)
library(TwoSampleMR)
library(MRInstruments)
library(data.table)
library(R.utils)
library(tidyverse)

#### 1.Exp:cis pQTL; Out:Finngen disease GWAS & UKB GWAS ####

type <- 'Incident'
Non_GWAS <- NULL
for (type in c('CrossSectional','Incident')){
  data <- read.xlsx(paste0("/public/home/dengyueting/Proteomics/Atlas/associations/results/result_merge_",type,"_All.xlsx"),sep=',')
  a <- as.data.frame(table(data$outcome))
  i=1
  for (i in 230:nrow(a)){
    print(i)
    disease <- as.character(a[i,1])
    dt <- subset(data,data$outcome == disease)
    pro <- as.data.frame(table(dt$Pro_code))
    pro <- pro[,1]
    IV <- fread("/public/home/dengyueting/Proteomics/Atlas/MR/cis_MR/data/cispQTL_IV_UKB.txt") %>% 
      filter(Pro_code %in% pro)
    IV <- as.data.frame(IV)
    if (nrow(IV) == 0){
      
    } else {
      
      IV <- IV[order(IV$P),]
      ## TwoSampleMR ##
      ins <- format_data(IV,type = "exposure",header = T,phenotype_col = "Pro_code",snp_col = "rsid",beta_col = "BETA",se_col = "SE",eaf_col = "A1FREQ",effect_allele_col = "ALLELE1",other_allele_col = "ALLELE0",pval_col = "P")
      tryCatch({
        pheno <- as.data.frame(fread(paste0("/public/home3/FinnGen-gwas/finngen_R9_",disease,".gz"),header = T))
        if (nrow(pheno) > 0){
          pheno$pheno <- disease
          out <- format_data(pheno,type = "outcome",header = T,snps = ins$SNP,phenotype_col = "pheno",snp_col = "rsids",beta_col = "beta",se_col = "sebeta",effect_allele_col ="alt",other_allele_col = "ref",eaf_col = "af_alt",pval_col = "pval")
          harmo <- harmonise_data(exposure_dat = ins,outcome_dat = out)
          mr_res <- mr(harmo,method_list = c("mr_wald_ratio","mr_ivw"))
          results_OR <- generate_odds_ratios(mr_res)
          results_OR <- results_OR[,-c(1,2)]
          write.table(results_OR,paste0("/public/home/dengyueting/Proteomics/Atlas/MR/cis_MR/results/",type,"/",disease,"_cis_MR.csv"),col.names = T,row.names = F,quote = F,sep = ",")
        } else {
          pheno <- as.data.frame(fread(paste0("/public/home2/nodecw_group/tmp/dyt_GCTA/results/geno_assoc_",disease,".fastGWA"),header = T))
          if (nrow(pheno) > 0){
            pheno$pheno <- disease
            out <- format_data(pheno,type = "outcome",header = T,snps = ins$SNP,phenotype_col = "pheno",snp_col = "SNP",beta_col = "BETA",se_col = "SE",effect_allele_col ="A1",other_allele_col = "A2",eaf_col = "AF1",pval_col = "P")
            harmo <- harmonise_data(exposure_dat = ins,outcome_dat = out)
            mr_res <- mr(harmo,method_list = c("mr_wald_ratio","mr_ivw"))
            results_OR <- generate_odds_ratios(mr_res)
            results_OR <- results_OR[,-c(1,2)]
            write.table(results_OR,paste0("/public/home/dengyueting/Proteomics/Atlas/MR/cis_MR/results/",type,"/",disease,"_cis_MR.csv"),col.names = T,row.names = F,quote = F,sep = ",")
          }
        }
      }, error = function(e){
        Non_GWAS <- c(Non_GWAS,disease)
      })
    }
    
    }
}
write.csv(Non_GWAS,"/public/home/dengyueting/Proteomics/Atlas/MR/cis_MR/results/non_GWAS_crs.txt")

array <- list.files("~/Proteomics/Atlas/MR/cis_MR/results/Incident//")



