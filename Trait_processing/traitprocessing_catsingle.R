
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
  make_option(c("-b", "--numParts"), type="character", default="ukb674930", 
              help="Should specify the names of ukb files (used to parellise)"),
  make_option(c("--fieldlist"), type="character", default="/public/home3/UKB_bulks/dataset_674930/ukb674930.csv", 
              help="ukbDir option should specify directory where ukb files were stored", metavar="character"),
  make_option(c("--array"), type="integer", default="1", 
              help="array to be processed", metavar="character"),
  make_option(c("-p", "--datafilepath"), type="character", default='/public/home/dengyueting/Proteomics/Atlas/phenotype/data/data-codes', 
              help="specify the data code files path", metavar="character"),
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
a <- as.data.frame(colnames(thisdata))
a <- colnames(thisdata)
a <- paste0('X',a)
a <- chartr('.','_', a)
a <- chartr('-','_', a)
print(paste0('Begin processing ',opt$numParts))
colnames(thisdata) <- a
colnames(thisdata)[1] <- 'eid'

#####CONTINOURS 先处理array=opt$array的基线数据#####
phenoInfo <- fread(paste0('/public/home/dengyueting/Proteomics/Atlas/phenotype/data/',opt$phenofile,opt$version,".tsv"))
phenoInfo <- subset(phenoInfo,phenoInfo$Array==opt$array)
phenoInfo <- subset(phenoInfo,phenoInfo$ValueType=='Categorical single')
## 质控phenoInfo
table(phenoInfo$newest3)
phenoInfo <- phenoInfo %>% filter( grepl(opt$numParts, newest3) ) %>% filter( !grepl('YES', EXCLUDED) ) %>% filter( !grepl('X', TRAIT_OF_INTEREST) )

## 用来计数以及存储,记下每个类别有多少个值，以及相应的colname;
pkg.env <- new.env(parent = emptyenv())
pkg.env$derivedBinary <- as.data.frame(thisdata[,"eid"])
pkg.env$derivedCont <- as.data.frame(thisdata[,"eid"])
pkg.env$derivedCatOrd <- as.data.frame(thisdata[,"eid"])
pkg.env$derivedCatUnord <- as.data.frame(thisdata[,"eid"])

#用到的function: 
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
getDataCode <- function(opt, varName) {
  # avoid annoying golbal binding from CRAN check
  dataPheno <- phenoInfo[which(phenoInfo$FieldID==varName),]
  dataCode <- dataPheno$DATA_CODING
  data_code_base <- file.path(dirname(opt$datafilepath),'data-codes')
  if (!file.exists(data_code_base)) {
    stop(paste0("Required dota codes folder ", data_code_base, " not found" ), call.=FALSE)
  }
  data_code_file <- file.path(data_code_base, paste0("datacode-", dataCode, ".tsv"))
  if (!file.exists(data_code_file)) {
    stop(paste0("Required dota codes file ", data_code_file, " not found" ), call.=FALSE)
  }
  datacode <- fread(data_code_file, sep='\t', header=TRUE, data.table=TRUE, na.strings=c("", "NA"))
  is.tree.code <- function(data_code_file) {
    datacode <- fread(data_code_file, sep='\t', header=TRUE, data.table=TRUE, na.strings=c("", "NA"))
    return ("parent_id" %in%  names(datacode));
  }
  annotate.tree.code <- function(data_code_file) {
    # avoid annoying golbal binding from CRAN check
    parent_id <- NULL
    level <- NULL
    meaning <- NULL
    node_id <- NULL
    root_node_id <- NULL
    
    #再次确认确实是一个tree code
    datacode <- fread(data_code_file, sep='\t', header=TRUE, data.table=TRUE, na.strings=c("", "NA"))
    if (! "parent_id" %in%  names(datacode)) {
      stop(paste0("parent_id column not found in ", data_code_file), call.=FALSE)
    }
    
    #parse tree structure，拆分树状结构，最终把分支合并到主干
    datacode <- datacode %>% mutate(meaning = gsub(",","|",meaning)) %>% 
      mutate(level = ifelse(parent_id ==0,0,NA)) %>% 
      mutate(root_node_id = ifelse(parent_id ==0,node_id,NA))
    all_done <- FALSE
    current_level <- 0
    while (!all_done) {
      parent_nodes <- datacode$node_id[which(datacode$level==current_level)]
      datacode <- datacode %>% mutate(level = ifelse(parent_id %in% parent_nodes, current_level+1, level))  
      for (pnode in parent_nodes) {
        root_row <- datacode %>% filter(node_id == pnode)
        root_value <- root_row$root_node_id
        datacode <- datacode %>%  mutate(root_node_id = ifelse(parent_id %in% pnode,root_value, root_node_id))
      }
      all_done <- !any(is.na(datacode$level))
      current_level <- current_level + 1
    }
    
    topNodes <- datacode %>% filter(parent_id == 0) %>% dplyr::select(root_node_id,meaning) %>% rename(root = meaning)
    
    datacode <- datacode %>% inner_join(topNodes)
    return(datacode)
    
  }
  if (is.tree.code(data_code_file)) {
    datacode <- annotate.tree.code(data_code_file)
  }
  return(datacode)
}
reorderOrderedCategory <- function(pheno,order) {
  ## new pheno of NAs (all values not in order are assumed to be NA)
  if (!is.na(order) && nchar(order)>0) {
    # make empty pheno		
    pheno2 <- rep(NA,nrow(thisdata))
    ## get ordering
    orderParts <- unlist(strsplit(order,"\\|"))
    
    # go through values in correct order and set value
    # from 1 to the number of values
    count <- 1
    for (i in orderParts) {
      idx <- which(pheno==i)
      pheno2[idx] <- count
      count <- count + 1
    }
    cat("reorder ",order," || ",sep="")
    return(pheno2)
  }	else {
    return(pheno)
  }
}
indicatorFields <- fread('/public/home/dengyueting/Proteomics/Atlas/phenotype/data/indicatorFields.tsv')
setDefaultValue <- function(pheno, defaultValue, defaultRelatedID, userID) {
  if (!is.na(defaultValue) && nchar(defaultValue)>0) {
    # remove people who have no value for indicator variable
    indName <- paste("X",defaultRelatedID,"_0_0",sep="")
    
    cat("Default related field: ", indName, " || ", sep="")
    userID <- thisdata[,1]
    indvarx <- merge(userID, indicatorFields, by="eid", all.x=TRUE, all.y=FALSE, sort=FALSE)
    indicatorVar <- indvarx[,indName,with=F]
    
    # check if there are already examples with default value and if so display warning
    numWithDefault <- length(which(pheno==defaultValue))
    if (numWithDefault>0) {
      cat("(WARNING: already ", numWithDefault, " values with default value) ", sep="")
    }
    
    # set default value in people who have no value in the pheno but do have a value in the default_value_related_field
    defaultIdxs <- which(!is.na(indicatorVar) & is.na(pheno))
    pheno[defaultIdxs] <- defaultValue
    cat("default value ", defaultValue, " set, N= ", length(defaultIdxs), " || ", sep="")
  }
  return(pheno)
}
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

#nrow(phenoInfo)

for (i in (1:nrow(phenoInfo))){
  varName <- phenoInfo$FieldID[i]
  print(paste0(i,"/",nrow(phenoInfo),"--",'Field: ',varName))
  varType <- phenoInfo$ValueType[i]
  colname <- paste0('X',varName,'_0_0')
  fit <- try(targetdata <- thisdata[,colname,with=F])
  if ("try-error" %in% class(fit)) {
    print(paste0('no ',colname,' in ',opt$numParts))
    next
  } else {
    targetdata <- thisdata[,colname,with=F]
    pheno <- targetdata
    datacode <- getDataCode(opt, varName)
    
    # reassignValue
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
    
    # get data code info - whether this data code is ordinal or not and any reordering
    dataPheno <- phenoInfo[which(phenoInfo$FieldID==varName),]
    dataCode <- dataPheno$DATA_CODING
    
    # get data coding information	
    dataCodeRow <- which(dataCodeInfo$dataCode==dataCode)
    if (length(dataCodeRow)==0) {
      cat("ERROR: No row in data coding info file || ")
      return
    }
    dataDataCode <- dataCodeInfo[dataCodeRow,]
    ordered <- dataDataCode$ordinal
    order <- as.character(dataDataCode$ordering)
    
    ## reorder variable values into increasing order (we do this now as this may convert variable to binary rather than ordered)
    pheno <- reorderOrderedCategory(pheno,order)
    
    ## if data code has a default_value then recode NA's to this value for participants with value in default_related_field
    ## this is used where there is no zero option e.g. field 100200
    defaultValue <- dataDataCode$default_value
    defaultRelatedID <- dataDataCode$default_related_field
    
    pheno <- setDefaultValue(pheno, defaultValue, defaultRelatedID)
    
    ## remove categories if < 10 examples
    pheno <- testNumExamples(opt, pheno)
    
    uniqVar <- as.data.frame(unique(na.omit(pheno)))
    if (nrow(uniqVar) <=2) {
    } else {
      uniqVar <- sort(uniqVar[,1])
    }
    
    ## Start saving the phenotype into different types
    if (length(uniqVar)<=1) { 
      cat("SKIP (only one value) || ")
      incrementCounter("catSin.fail.onevalue")
    } else if (length(uniqVar)==2) {		
      cat("CAT-SINGLE-BINARY || ")
      phenoFactor <- factor(pheno)
      # check sample size
      numNotNA <- length(which(!is.na(phenoFactor)))
      if (numNotNA < opt$binnacutoff) {
        cat("BIN-SKIP-", opt$binnacutoff, " (", numNotNA, ") || ",sep="")
        incrementCounter(paste("catSin.binary.fail.N", opt$catordnacutoff, sep=""))
      } else {
        incrementCounter("catSin.binary.success")
        # increase binary pheno data
        storeNewVar(thisdata[,"eid"], phenoFactor, colname, 'bin')
      }
    } else {
      # > 2 categories
      if (is.na(ordered)) {
        cat(" ERROR: 'ordered' not found in data code info file")	
      }  else {
        ## unordered
        if (ordered == 0) {
          cat("CAT-SINGLE-UNORDERED || ")
          numNotNA <- length(which(!is.na(pheno)))
          if (numNotNA < opt$catunordnacutoff) {
            cat("CATUNORD-SKIP-", opt$catunordnacutoff, " (", numNotNA, ") || ",sep="")
            incrementCounter(paste("catSin.unordCat.fail.N", opt$catunordnacutoff, sep=""))
          } else {
            # check there are not too many levels and skip if there are
            numUnique <- nrow(as.data.frame(unique(na.omit(pheno))))
            if (numUnique > opt$maxunorderedcategories) {
              cat("Too many levels: ", numUnique, " > ", opt$maxunorderedcategories, 
                  "(num outcomes values: ", numUnique, ") || SKIP ", sep="")
              incrementCounter("catSin.unordCat.fail.manyvalues")
              return
            }
            incrementCounter("catSin.unordCat.success.need_reorder")
            storeNewVar(thisdata[,"eid"],pheno, colname, 'catUnord')
          }
        } else if (ordered == -1) {
          ## ordered
          cat("ordered || ")
          
          # log the ordering of categories used
          .setOrderString <- function(orderStr, uniqVar) {
            if (is.na(orderStr) || nchar(orderStr)==0) {
              orderStr <- ""
              # create order str by appending each value
              uniqVarSorted <- sort(uniqVar)
              first <- TRUE
              for (i in uniqVarSorted) {
                if (!first) {
                  orderStr <- paste(orderStr, "|",	sep="")
                }
                if (i >= 0)  {# ignore missing values
                  orderStr <- paste(orderStr, i, sep="")
                }
                first <- FALSE
              }
            }
            return(orderStr)
          }
          orderStr <-.setOrderString(order, uniqVar)
          cat("order: ", orderStr, " || ",  sep="")
          # check sample size
          numNotNA <- length(which(!is.na(pheno)))
          if (numNotNA < opt$catordnacutoff) {
            cat("CATORD-SKIP-", opt$catordnacutoff, " (", numNotNA, ") || ",sep="")
            incrementCounter(paste("catSin.ordCat.fail.N", opt$catordnacutoff, sep=""))
          } else {
            #phenoFactor <- factor(pheno)
            cat("num categories: ", nrow(unique(na.omit(pheno))), " || ", sep="");
            incrementCounter("catSin.ordCat.success")
            storeNewVar(thisdata[,"eid"], pheno, colname, 'catOrd')
          }
          
        } else if (ordered == -2) {
          cat(" EXCLUDED or BINARY variable: Should not get here in code. ")
          incrementCounter( "catSin.binary.or.excluded.fail")
        } else {
          print(paste("ERROR", varName, varType, dataCode));
        }
      }
    }
  }
  print(' ')
}

sourcedata <- opt$numParts
varType <- gsub(" ","_",varType)
write.csv(pkg.env[["counters"]],paste0(opt$resDir,"/",opt$version,"/",sourcedata,"_",varType,"_conters.csv"))
write.table(pkg.env$derivedBinary,paste0(opt$resDir,"/",opt$version,"/",sourcedata,"_",varType,"_derivedBinary.txt"),col.names = T,row.names = F,sep = "\t",quote = F)
write.table(pkg.env$derivedCont,paste0(opt$resDir,"/",opt$version,"/",sourcedata,"_",varType,"_derivedCont.txt"),col.names = T,row.names = F,sep = "\t",quote = F)
write.table(pkg.env$derivedCatOrd,paste0(opt$resDir,"/",opt$version,"/",sourcedata,"_",varType,"_derivedCatOrd.txt"),col.names = T,row.names = F,sep = "\t",quote = F)
write.table(pkg.env$derivedCatUnord,paste0(opt$resDir,"/",opt$version,"/",sourcedata,"_",varType,"_derivedCatUnord.txt"),col.names = T,row.names = F,sep = "\t",quote = F)

print(paste0('Finish processing ',opt$numParts))

