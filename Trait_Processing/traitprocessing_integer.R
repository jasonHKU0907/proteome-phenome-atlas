# Processing integer fields, namely:
# 1) Reassigning values as specified in the data code information file
# 2) Treating this field as continuous if at least 20 distinct values.
# Otherwise treat as binary or ordered categorical if 2 or more than two values. 

library(data.table)
library(PEACOK)
library(dplyr)
library('optparse')
option_list = list(
  make_option(c("-v", "--version"), type="character", default="v1203", 
              help="the name of the update of UKB pheno (for update use)", metavar="character"),
  make_option(c("-f", "--phenofile"), type="character", default="outcome_info_new_", 
              help="Phenotype dataset file name", metavar="character"),
  make_option(c("-d", "--datacodingfile"), type="character", default='data_coding_ordinal_info_new_', 
              help="datacodingfile file name (should be comma separated)", metavar="character"),
  make_option(c("-b", "--numParts"), type="character", default="subset_1", 
              help="Should specify the names of ukb files (used to parellise)"),
  make_option(c("--fieldlist"), type="character", default="/public/mig_old_storage/home2/UKB_Tabular_merged_10/UKB_subset_1.csv", 
              help="ukbDir option should specify directory where ukb files were stored", metavar="character"),
  make_option(c("--array"), type="integer", default="1", 
              help="array to be processed", metavar="character"),
  make_option(c("-e", "--traitofinterest"), type="character", default=NULL, 
              help="traitofinterest option should specify trait of interest variable name", metavar="character"),
  make_option(c("-r", "--resDir"), type="character", default="/public/home/dengyueting/Proteomics/Atlas/phenotype/results", 
              help="resDir option should specify directory where results files should be stored", metavar="character"),
  make_option(c("-u", "--userId"), type="character", default="eid", 
              help="userId option should specify user ID column in trait of interest and phenotype files [default= %default]", metavar="character"),
  
  make_option(c("--catmultcutoff"), type="integer", default=20, 
              help="The cutoff for exclusion when creating dichotomous variables for CAT MULTIPLE."),
  make_option(c("--catordnacutoff"), type="integer", default=1000,
              help="The cutoff for exclusion for number of non NAs in ordered categorical variables."),
  make_option(c("--catunordnacutoff"), type="integer", default=1000,
              help="The cutoff for exclusion for number of non NAs in unordered categorical variables."),
  make_option(c("--contnacutoff"), type="integer", default=1000,
              help="The cutoff for exclusion for number of non NAs in continuous variables."),
  make_option(c("--binnacutoff"), type="integer", default=1000,
              help="The cutoff for exclusion for number of non NAs in binary variables."),
  make_option(c("--bintruecutoff"), type="integer", default=20,
              help="The cutoff for exclusion for numbers of members of a category in binary variables."),
  make_option(c("--mincategorysize"), type="integer", default=20,
              help="The minimum number of samples in a category."),
  make_option(c("--maxunorderedcategories"), type="integer", default=1000,
              help="The maximum number of categories in an unordered categorical variable"),
  make_option(c("--propforcontinuous"), type="double", default=0.2,
              help="The cutoff for proportion of samples with the same value for the variable to not be considered continuous.")
)

opt_parser = OptionParser(option_list=option_list)  #解析参数
opt = parse_args(opt_parser)
dataCodeInfo <- fread(paste0('/public/home/dengyueting/Proteomics/Atlas/phenotype/data/',opt$datacodingfile,opt$version,".txt"))

thisdata <- fread(opt$fieldlist, na.strings=c("", "NA"))
a <-as.data.frame(colnames(thisdata))
a <- colnames(thisdata)
a <- paste0('X',a)
a <- chartr('.','_', a)
a <- chartr('-','_', a)
#a
colnames(thisdata) <- a
colnames(thisdata)[1] <- 'eid'

#####INTEGER 先处理array=1的基线数据#####
phenoInfo <- fread(paste0('/public/home/dengyueting/Proteomics/Atlas/phenotype/data/',opt$phenofile,opt$version,".tsv"))
phenoInfo <- subset(phenoInfo,phenoInfo$Array==opt$array)
phenoInfo <- subset(phenoInfo,phenoInfo$ValueType=='Integer')
## 质控phenoInfo
#table(phenoInfo$newest3)
phenoInfo <- phenoInfo %>% filter( grepl(opt$numParts, newest3) ) %>% filter( !grepl('YES', EXCLUDED) ) %>% filter( !grepl('X', TRAIT_OF_INTEREST) )

## 用来计数以及存储,记下每个类别有多少个值，以及相应的colname;
pkg.env <- new.env(parent = emptyenv())
pkg.env$derivedBinary <- as.data.frame(thisdata[,"eid"])
pkg.env$derivedCont <- as.data.frame(thisdata[,"eid"])
pkg.env$derivedCatOrd <- as.data.frame(thisdata[,"eid"])
pkg.env$derivedCatUnord <- as.data.frame(thisdata[,"eid"])
# 计数
incrementCounter <- function(countName) {
  idx = which(pkg.env$counters$name==countName)
  if (length(idx)==0) {
    # counter does not exist so add with countValue 1
    pkg.env$counters <- rbind(pkg.env$counters, data.frame(name=countName, countValue=1,fieldID=colname))
  } else {
    # increment counter that already exists
    pkg.env$counters$countValue[idx] <- pkg.env$counters$countValue[idx]+1
    pkg.env$counters$fieldID[idx] <- paste0(pkg.env$counters$fieldID[idx],",",colname)
  }
}
# 保存文件
storeNewVar <- function(userIDData, phenoData, colName, type) {
  # add pheno to dataframe
  newdata = data.frame(userID=userIDData, newvar=phenoData)
  names(newdata)[names(newdata)=="newvar"] = colname
  if (type == "bin") {
    pkg.env$derivedBinary <- merge(pkg.env$derivedBinary, newdata, by="eid", all=TRUE)
  } else if (type == "cont") {
    pkg.env$derivedCont <- merge(pkg.env$derivedCont, newdata, by="eid", all=TRUE)
  } else if (type == "catOrd") {
    pkg.env$derivedCatOrd <- merge(pkg.env$derivedCatOrd, newdata, by="eid", all=TRUE)
  } else if (type == "catUnord") {
    pkg.env$derivedCatUnord <- merge(pkg.env$derivedCatUnord, newdata, by="eid", all=TRUE)
  }
}
replaceNaN <- function(pheno) {
  if (is.factor(pheno)) {
    phenoReplaced <- pheno
    nanStr  <- which(phenoReplaced=="NaN")
    phenoReplaced[nanStr] <- NA 
    emptyx  <- which(phenoReplaced=="")
    phenoReplaced[emptyx] <- NA
  } else {
    phenoReplaced <- pheno
    nanx  <-  which(is.nan(phenoReplaced))
    phenoReplaced[nanx] <- NA
    emptyStr  <- which(phenoReplaced=="")	
    phenoReplaced[emptyStr] <- NA
  }
  return(phenoReplaced)
}
# 将连续型数值分为3个或2个bin
.equalSizedBins <- function(phenoAvg) {
  ## equal sized bins 
  q <- quantile(phenoAvg, probs=c(1/3,2/3), na.rm=TRUE)
  
  minX <- min(phenoAvg, na.rm=TRUE)
  maxX <- max(phenoAvg, na.rm=TRUE)
  
  phenoBinned <- phenoAvg
  if (q[1]==minX) {
    # edge case - quantile value is lowest value
    # assign min value as cat1
    idx1 <- which(phenoAvg==q[1])
    phenoBinned[idx1] <- 0
    
    # divide remaining values into cat2 and cat3
    phenoAvgRemaining <- phenoAvg[which(phenoAvg!=q[1])]
    qx <- quantile(phenoAvgRemaining, probs=c(0.5), na.rm=TRUE)
    minXX <- min(phenoAvgRemaining, na.rm=TRUE)
    maxXX <-	max(phenoAvgRemaining, na.rm=TRUE)
    
    if (qx[1]==minXX) {
      # edge case again - quantile value is lowest value
      idx2 <- which(phenoAvg==qx[1])
      idx3 <- which(phenoAvg>qx[1])
    } else if (qx[1]==maxXX) {
      # edge case again - quantile value is max value
      idx2 <- which(phenoAvg<qx[1] & phenoAvg>q[1])
      idx3 <- which(phenoAvg==qx[1])
    } else {
      idx2 <- which(phenoAvg<qx[1] & phenoAvg>q[1])
      idx3 <- which(phenoAvg>=qx[1])
    }
    phenoBinned[idx2] <- 1
    phenoBinned[idx3] <- 2
  } else if (q[2]==maxX) {
    # edge case - quantile value is highest value
    # assign max value as cat3
    idx3 <- which(phenoAvg==q[2])
    phenoBinned[idx3] <- 2
    
    # divide remaining values into cat1 and cat2
    phenoAvgRemaining <- phenoAvg[which(phenoAvg!=q[2])]
    qx <- quantile(phenoAvgRemaining, probs=c(0.5), na.rm=TRUE)
    minXX <- min(phenoAvgRemaining, na.rm=TRUE)
    maxXX <- max(phenoAvgRemaining, na.rm=TRUE)
    
    if (qx[1]==minXX) {
      # edge case again - quantile value is lowest value
      idx1 <- which(phenoAvg==qx[1])
      idx2 <- which(phenoAvg>qx[1] & phenoAvg<q[2])
    } else if	(qx[1]==maxXX) {
      # edge case again - quantile value is max value
      idx1 <- which(phenoAvg<qx[1])
      idx2 <- which(phenoAvg==qx[1])
    } else {
      idx1 <- which(phenoAvg<qx[1])  
      idx2 <- which(phenoAvg>=qx[1] & phenoAvg<q[2])
    }
    
    phenoBinned[idx1] <- 0
    phenoBinned[idx2] <- 1
  } else if (q[1] == q[2]) {
    # both quantiles correspond to the same value so set 
    # cat1 as < this value, cat2 as exactly this value and
    # cat3 as > this value
    phenoBinned <- phenoAvg
    idx1 <- which(phenoAvg<q[1])
    idx2 <- which(phenoAvg==q[2])
    idx3 <- which(phenoAvg>q[2])
    phenoBinned[idx1] <- 0
    phenoBinned[idx2] <- 1
    phenoBinned[idx3] <- 2
  } else {
    # standard case - split the data into three roughly equal parts where
    # cat1<q1, cat2 between q1 and q2, and cat3>=q2
    phenoBinned <- phenoAvg
    idx1 <- which(phenoAvg<q[1])
    idx2 <- which(phenoAvg>=q[1] & phenoAvg<q[2])
    idx3 <- which(phenoAvg>=q[2])
    phenoBinned[idx1] <- 0
    phenoBinned[idx2] <- 1
    phenoBinned[idx3] <- 2
  }
  
  cat("cat N: ", length(idx1),", ",length(idx2),", ",length(idx3), " || ", sep="")
  return(phenoBinned)
}

for (i in (1:nrow(phenoInfo))){
  print(paste0(i,'/',nrow(phenoInfo)))
  varName <- phenoInfo$FieldID[i]
  varType <- phenoInfo$ValueType[i]
  colname <- paste0('X',varName,'_0_0')
  # 测试该UKB数据集是否有该field ID的数据
  fit <- try(targetdata <- thisdata[,colname,with=F])
  if ("try-error" %in% class(fit)) {
    print(paste0('no ',colname,' in ',opt$ukbfile))
    next
  } else {
    targetdata <- thisdata[,colname,with=F]
    pheno <- targetdata
    cat("INTEGER || ")
    if (is.numeric(as.matrix(pheno))) {
      # 根据dataCodeInfo将原始数编码，去除无意义值、分等级etc
      reassignValue <- function(pheno, varName) {
        # get data code info - whether this data code is ordinal or not and any reordering and resassignments
        dataPheno <- phenoInfo[which(phenoInfo$FieldID==varName),]
        dataCode <- dataPheno$DATA_CODING
        
        # not all variables will have a data code info row
        dataCodeRow <- which(dataCodeInfo$dataCode==dataCode)
        if (length(dataCodeRow)==0) { #do nothing 
        } else if (length(dataCodeRow)==1) {
          dataDataCode <- dataCodeInfo[dataCodeRow,]
          reassignments <- as.character(dataDataCode$reassignments)
          
          # Reassigns values in pheno, as specified in resassignments argument
          # can be NA if row not included in data coding info file
          if (!is.na(reassignments) && nchar(reassignments)>0) {
            reassignParts <- unlist(strsplit(reassignments,"\\|"))
            cat(paste("reassignments: ", reassignments, " || ", sep=""))
            
            # do each reassignment
            for(i in reassignParts) {
              current <- unlist(strsplit(i,"="))
              
              # matrix version
              if (is.na(strtoi(current[1]))) {
                idx <- which(is.na(pheno), arr.ind=TRUE)
              } else {
                idx <- which(pheno==current[1],arr.ind=TRUE)
              }
              pheno[idx] <- strtoi(current[2])
            }
            
            ## see if type has changed (this happens for field 216 (X changed to -1))
            ## as.numeric will set non numeric to NA so we know if it's ok to do this by seeing if there are extra NA's after the conversion
            pNum = as.numeric(unlist(pheno))
            isNum = length(which(is.na(pheno), arr.ind=TRUE))==length(which(is.na(pNum), arr.ind=TRUE))
            if (isNum) {
              pheno = pNum
            }
          }
        } else {
          cat("WARNING: >1 ROWS IN DATA CODE INFO FILE || ")
          
        }
        return(pheno)
      }
      pheno <- reassignValue(pheno, varName)
      ## average if multiple columns（我们取的是array=1的值，不进行这一步也行）
      if (!is.null(dim(pheno))) {
        phenoAvg <- rowMeans(pheno, na.rm=TRUE)
        # if participant only has NA values then NaN is generated so we convert back to NA
        phenoAvg <- replaceNaN(phenoAvg)
      } else {
        phenoAvg <- pheno
      }
      
      uniqVar <- unique(na.omit(phenoAvg))
      # if >=20 separate values then treat as continuous variable
      if (length(uniqVar)>=20) {
        ## whether > opt$propforcontinuous examples with same value
        ## 检验是否有这种连续型变量不均匀分布的情况
        uniqVar <- unique(na.omit(phenoAvg))
        valid <- TRUE
        for (uniq in uniqVar) {
          numWithValue <- length(which(phenoAvg==uniq))
          numNotNA <- length(which(!is.na(phenoAvg)))
          if (numWithValue / numNotNA >= opt$propforcontinuous) {
            valid <- FALSE
            break #（终止循环语句并将执行转移到循环之后的语句）
          }
        }
        if (!valid) {
          ## 如果>20%的人拥有同样的值，treat as ordinal categorical
          cat(">", opt$propforcontinuous*100, "% IN ONE CATEGORY || ")
          ## try to treat as ordered categorical
          incrementCounter("int.conttry.ordcattry")
          ## equal sized bins
          phenoBinned <- .equalSizedBins(phenoAvg)
          
          # check number of people in each bin
          bin0Num <- length(which(phenoBinned==0))
          bin1Num <- length(which(phenoBinned==1))
          bin2Num <- length(which(phenoBinned==2))
          
          if (bin0Num>=10 & bin1Num>=10 & bin2Num>=10) {
            # successful binning. >=10 examples in each of the 3 bins
            # check sample size
            numNotNA <- length(which(!is.na(phenoBinned)))
            if (numNotNA < opt$catordnacutoff) {
              cat("CATORD-SKIP-", opt$catordnacutoff, " (", numNotNA, ") || ",sep="")
              incrementCounter(paste("int.conttry.ordcattry.fail.N.", opt$catordnacutoff, sep=""))
            } else {
              # increase ordered cattry pheno data	
              phenoFactor <- factor(phenoBinned)
              cat("num categories: ", length(unique(na.omit(phenoFactor))), " || ", sep="")
              incrementCounter("int.conttry.ordcattry.success")
              storeNewVar(thisdata[,"eid"], phenoBinned, colname, 'catOrd')
            }
          } else {
            # try to treat as binary because not enough examples in each bin
            if (bin0Num<10 & bin2Num<10) {
              ## skip - not possible to create binary variable because first and third bins are too small
              ## ie. could merge bin1 with bin 2 but then bin3 still too small etc
              cat("SKIP 2 bins are too small || ")
              incrementCounter("int.conttry.ordcattry.fail.smallbins1")
            } else if ((bin0Num<10 | bin1Num<10) & (bin0Num+bin1Num)>=10) {
              # combine first and second bin to create binary variable
              cat("Combine first two bins and treat as binary || ")
              phenoBinned[which(phenoBinned==0)] <-1	
              # check sample size
              numNotNA <- length(which(!is.na(phenoBinned)))
              if (numNotNA < opt$binnacutoff) {
                cat("BIN-SKIP-", opt$binnacutoff, " (", numNotNA, ") || ",sep="")
                incrementCounter(paste("int.conttry.ordcattry.binsbinary.fail.N.", opt$catordnacutoff, sep=""))
              } else {
                incrementCounter("int.conttry.ordcattry.binsbinary.success")
                # increase binary pheno data
                storeNewVar(thisdata[,"eid"], phenoBinned, colname, 'bin')
              }
            } else if ((bin2Num<10 | bin1Num<10) & (bin2Num+bin1Num)>=10) {
              # combine second and last bin to create binary variable
              cat("Combine last two bins and treat as binary || ")
              phenoBinned[which(phenoBinned==2)] <- 1
              # check sample size
              numNotNA <- length(which(!is.na(phenoBinned)))
              if (numNotNA < opt$binnacutoff) {
                cat("BIN-SKIP-", opt$binnacutoff, " (", numNotNA, ") || ",sep="")
                incrementCounter(paste("int.conttry.ordcattry.binsbinary.fail.N.", opt$catordnacutoff, sep=""))
              } else {
                incrementCounter("int.conttry.ordcattry.binsbinary.success")
                # increase binary pheno data
                storeNewVar(thisdata[,"eid"], phenoBinned, colname, 'bin')
              }
            } else {
              ## skip - not possible to create binary variable because combining bins would still be too small
              cat("SKIP 2 bins are too small(2) || ")
              incrementCounter("int.conttry.ordcattry.fail.smallbins2")
            }
          }
        } else {
          ## 如果拥有同样的值的人数<20%，可以treat as continuous
          cat("non IRNT || ")
          # check there are at least 500 examples
          numNotNA <- length(which(!is.na(phenoAvg)))
          if (numNotNA < opt$contnacutoff) {
            cat("INTEGER-SKIP-", opt$contnacutoff, " (", numNotNA, ") || ",sep="")
            incrementCounter(paste("int.conttry.fail.N", opt$contnacutoff, sep=""))
          } else {
            storeNewVar(thisdata[,"eid"], phenoAvg, colname, 'cont')
            cat("SUCCESS save-interger","-",i)
            incrementCounter("int.conttry.success")
          }
        }
      } else {
        ## remove categories if < 10 examples
        testNumExamples <- function(opt, pheno) {
          ## Loop through values and remove if has < opt$mincategorysize examples
          uniqVar <- unique(na.omit(pheno))
          for (u in uniqVar) {
            withValIdx <- which(pheno==u)
            numWithVal <- length(withValIdx);
            if (numWithVal < opt$mincategorysize) {
              pheno[withValIdx] <- NA
              cat("Removed ",u ,": ", numWithVal, "<", opt$mincategorysize, " examples || ", sep="");
            } else {
              cat("Inc(>=", opt$mincategorysize, "): ", u, "(", numWithVal, ") || ", sep="");
            }
          }
          return(pheno)
        }        
        phenoAvg <- testNumExamples(opt, phenoAvg)
        ## binary if 2 distinct values, else ordered categorical
        phenoFactor <- factor(phenoAvg)
        numLevels <- length(levels(phenoFactor))
        if (numLevels<=1) {
          cat("SKIP (number of levels: ",numLevels,")",sep="")
          incrementCounter("int.cattry.onevalue")
        } else if (numLevels==2) {
          # binary
          # check sample size
          numNotNA <- length(which(!is.na(phenoFactor)))
          if (numNotNA < opt$binnacutoff) {
            cat("BIN-SKIP-", opt$binnacutoff, " (", numNotNA, ") || ",sep="")
            incrementCounter(paste("int.cattry.binary.fail.N", opt$catordnacutoff, sep=""))
          } else {
            incrementCounter("int.cattry.binary.sucess")
            # increase binary pheno data
            storeNewVar(thisdata[,"eid"], phenoFactor, colname, 'bin')
          }
        } else {
          cat("3-20 values || ")
          # we don't use equal sized bins just the original integers (that have >=10 examples) as categories
          ## testNumExamples已经测试每个都have >=10 examples
          # check sample size
          numNotNA <- length(which(!is.na(phenoFactor)))
          if (numNotNA < opt$binnacutoff) {
            cat("BIN-SKIP-", opt$binnacutoff, " (", numNotNA, ") || ",sep="")
            incrementCounter(paste("int.ordcattry.fail.N", opt$catordnacutoff, sep=""))
          } else {
            incrementCounter("int.ordcattry.success")
            # increase ordinary categorical pheno data
            storeNewVar(thisdata[,"eid"], phenoFactor, colname, 'catOrd')
          }
        }
      }
    } else {
      cat("SKIP Integer type but not numeric",sep="")
    }
    
  }
  print(i)
}

sourcedata <- opt$numParts
write.csv(pkg.env[["counters"]],paste0(opt$resDir,"/",opt$version,"/",sourcedata,"_",varType,"_conters.csv"))
write.table(pkg.env$derivedBinary,paste0(opt$resDir,"/",opt$version,"/",sourcedata,"_",varType,"_derivedBinary.txt"),col.names = T,row.names = F,sep = "\t",quote = F)
write.table(pkg.env$derivedCont,paste0(opt$resDir,"/",opt$version,"/",sourcedata,"_",varType,"_derivedCont.txt"),col.names = T,row.names = F,sep = "\t",quote = F)
write.table(pkg.env$derivedCatOrd,paste0(opt$resDir,"/",opt$version,"/",sourcedata,"_",varType,"_derivedCatOrd.txt"),col.names = T,row.names = F,sep = "\t",quote = F)
write.table(pkg.env$derivedCatUnord,paste0(opt$resDir,"/",opt$version,"/",sourcedata,"_",varType,"_derivedCatUnord.txt"),col.names = T,row.names = F,sep = "\t",quote = F)



