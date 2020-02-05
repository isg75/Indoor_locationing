
## Project: Wifi Locationing
## Author: Matias Barra

## Script purpose: use the signal intensity recorded from multiple wifi hotspots 
## within the building to determine users location using Machine Learning Models.

####------------------------------------------------ Set Environment --------------------####

#Load required libraries 
pacman:: p_load("rstudioapi", "readr","dplyr", "tidyr", "ggplot2", "plotly", "data.table", "reshape2", "caret", 
                "randomForest", "doMC", "bbplot", "parallel", "iterators", "ranger", "tidyverse", "class",
                "doParallel", "arules","ggthemes","devtools","magrittr")

# Import Data
#### Ignacio: Matias, avoid making use of this approach as it is computer dependant
#### In other words, this will not work into another computer as the new computer
#### doesn't need to have the same OS and same folder tree.
#setwd("/Users/matiasbarra/Documents/Data_Analytics_Course/3_IoT_Analytics/3.3_Wifi_locationing")
#### You can use the function "list.files", and then use the function "grep"
#### to look for your script.

loc   <- grep("data_exploration.R",list.files(recursive=TRUE),value=TRUE)
iloc  <- which(unlist(gregexpr("/data_exploration.R$",loc)) != -1)
myloc <- paste(getwd(),loc[iloc],sep="/")
setwd(substr(myloc,1,nchar(myloc)-18))

#data_train <- read.csv("WiFi_Locationing/data/trainingData.csv", header = T, sep = ",", na='100')
#data_validation <- read.csv("WiFi_Locationing/data/validationData.csv", header = T, sep = ",", na='100')

#### Now we're in the "src", we need to read one level up.
data_train <- read.csv("../data/trainingData.csv", header = T, sep = ",", na='100')
data_validation <- read.csv("../data/validationData.csv", header = T, sep = ",", na='100')

# Check if we have the same names of variables      #<-Yes, we can merge both data sets
"%ni%" <- Negate("%in%")
#names(data_train[which(names(data_train) %ni% names(data_train))])   #<-0
#names(data_validation[which(names(data_validation) %ni% names(data_validation))])   #<-0

# Use distinct to Remove repeated rows in data_train and data_validation             
data_train <- dplyr::distinct(data_train)       #<- 19937 to 19300
data_validation <- dplyr::distinct(data_validation)       #<- No repeated rows!!

# #add variable type, to create only one df and identify train and test
data_train$type <- "train"
data_validation$type <- "test"

#make the dataset 1 set
data_full <- bind_rows(data_train, data_validation)

#### Changing Valiables Format ####

#### Preprocessing
pre_proc <- function(df) {
  
  #Changing timestamp variables to POSIXct
  df$TIMESTAMP <- as.POSIXct(df$TIMESTAMP, origin = "1970-01-01")
  
  # Changing timestamp variables to factor using DPLYR (More eficiently and faster)
  vars_to_factor <- c("FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID")
  vars_to_numeric <- c("LONGITUDE", "LATITUDE")
  
  df %<>% mutate_each_(funs(factor), vars_to_factor) %>%
          mutate_each_(funs(as.numeric), vars_to_numeric)
  
  #   for (v in vars_to_factor) {
  #   df[,v] <- as.factor(df[,v])
  # }
  # rm(vars_to_factor)
  
  # # Changing timestamp variables to factor using a lapply funtion
  # vars_to_numeric <- c("LONGITUDE", "LATITUDE")
  # df[,vars_to_numeric] <- lapply(df[,vars_to_numeric], as.numeric)
  # rm(vars_to_numeric)
  
  # Use Paste function to create an ID for BuildingID + Floor and convert the new column to factor
  df <-  within(df, bfID <- paste(BUILDINGID, FLOOR, sep = '_'))
  df$bfID <- as.factor(df$bfID)
  
  # Deleting columns and rows wich contain only NAs
  df <- df[-which(apply(df[,c(1:520)],1,function(x) all(is.na(x)==T))==TRUE),]
  
  # Add count of WAP's detected as feature
  df$WAP_num <- apply(df[,1:520], 1,
                             function(x) length(which(!is.na(x))))

  # Using regular expression
  WAPS <- colnames(df)[which(unlist(gregexpr("^WAP\\d+",colnames(df)))==1)]
  
  #WAPS <- grep("WAP", colnames(df), value=T) #value = TRUE returns the values instead of the vectors

  # # Create new variables HighWAP and HighRSSI
  # df <- df %>%
  #   mutate(HighWAP=NA)
  
  # # Gets which columns has the highest value for a fingerprint
  # df <- df %>% 
  #   mutate(HighWAP=colnames(df[WAPS])[apply(df[WAPS],1,which.max)])
  
  # Function to find 1st, 2nd, 3rd highest WAPs and RSSI values 
  maxn <- function(n) function(x) order(x, decreasing = TRUE)[n]
  
  # # Set 1st highest WAP number
  # df <- df %>% 
  #   mutate(HighWAP=colnames(df[WAPS])[apply(df[,WAPS],1,which.max)])
  
  df %<>% 
     mutate(HighWAP=colnames(df[WAPS])[apply(df[WAPS],1,maxn(1))],
            HighWAP2=colnames(df[WAPS])[apply(df[WAPS],1,maxn(2))],
            HighWAP3=colnames(df[WAPS])[apply(df[WAPS],1,maxn(3))],
            HighWAP4=colnames(df[WAPS])[apply(df[WAPS],1,maxn(4))])
  
  # # Set 2nd highest WAP number
  # df %>% 
  #   mutate(HighWAP2=colnames(df[WAPS])[apply(df[WAPS],1,maxn(2))])
  # 
  # # Set 3rd highest WAP number
  # df %>% 
  #   mutate(HighWAP3=colnames(df[WAPS])[apply(df[WAPS],1,maxn(3))])
  # 
  # # Set 4th highest WAP number
  # df %>% 
  #   mutate(HighWAP4=colnames(df[WAPS])[apply(df[WAPS],1,maxn(4))])
  
  # convert NA's to -110
  df[is.na(df)] <- -110
  
  vars_to_factor <- c("HighWAP", "HighWAP2", "HighWAP3", "HighWAP4")
  
  df %<>% mutate_each_(funs(factor), vars_to_factor) 
  
  # for (v in vars_to_factor) {
  #   #  data_full[,v] <- as.factor(data_full[,v])
  #   df[,v] <- factor(df[,v],levels = colnames(df[,1:520]),
  #                           labels = colnames(df[,1:520]))
  # }
  # rm(vars_to_factor)
  
  # reorder columns 
  last_col <- c(521:530)
  df <-  df %>% select(last_col, everything())
  rm(last_col)
  
  return(df)
}

data_full <- pre_proc(data_full)

# data_full$TIMESTAMP <- as.POSIXct(data_full$TIMESTAMP, origin = "1970-01-01")
# 
# #-Recode Building factor level names
# #### Ignacio: Very dangerous thing to do as you already know ;)
# #data_full$BUILDINGID <- recode(data_full$BUILDINGID, '0'=1, '1'=2, '2'=3)
# 
# #Changing timestamp variables to factor using a For
# vars_to_factor <- c("FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID")
# for (v in vars_to_factor) {
#   data_full[,v] <- as.factor(data_full[,v])
# }
# rm(vars_to_factor)
# 
# #Changing timestamp variables to factor using a lapply funtion
# vars_to_numeric <- c("LONGITUDE", "LATITUDE")
# data_full[,vars_to_numeric] <- lapply(data_full[,vars_to_numeric], as.numeric)
# rm(vars_to_numeric)
# 
# # Use Paste function to create an ID for BuildingID + Floor and convert the new column to factor
# data_full <-  within(data_full, bfID <- paste(BUILDINGID, FLOOR, sep = '_'))
# data_full$bfID <- as.factor(data_full$bfID)
# 
# #Deleting columns and rows wich contain only NAs
# data_full <- data_full[-which(apply(data_full[,c(1:520)],1,function(x) all(is.na(x)==T))==TRUE),]
# #Only delete 73 rows, there is no columns containig all NAs
# 
# # Add count of WAP's detected as feature
# data_full$WAP_num <- apply(data_full[,1:520], 1,
#                            function(x) length(which(!is.na(x))))  

####------------------------------------------------ Data Exploration --------------------####

#Wifi Measurements - FULL SET
plot_ly(data_full, x= ~LONGITUDE,y= ~LATITUDE,z= ~FLOOR, 
              type="scatter3d", mode="markers", size = 150,
              color = ~type, colors = c('red', 'blue')) %>%
  layout(title = 'Latitude and Longitude for each observation',
         scene = list(
           xaxis = list(title = 'LONGITUDE'),
           yaxis = list(title = 'LATITUDE',tickangle = 180,gridcolor = 'grey50',zerolinewidth = 1,
                        range = c(4864747,4865018),zeroline = FALSE,showline = FALSE,showticklabels = FALSE),
           zaxis = list(title = 'Floors', gridcolor = 'grey50', zerolinewidth = 1, ticklen = 5,
                        gridwith = 2,categoryorder = "array",categoryarray = c("1st Floor", "2nd Floor", 
                                          "3rd Floor","4th Floor","5th Floor"))),
         paper_bgcolor = 'rgb(243, 243, 243)',
         plot_bgcolor = 'rgb(243, 243, 243)')


#Box Plots 
#-Distribution of WAP count by building- boxplot
#### Ignacio: Cool plot, this tells you that the coverage is not uniform.
box_build <-  ggplot(data_full, aes(x=BUILDINGID, y=WAP_num)) + 
                geom_boxplot(fill='lightgreen') +
                theme(text = element_text(size=14)) +
                ggtitle('Distribution of Detected Wireless Access Points by Building') +
                labs(x="Building Number", y= 'WAP Counts' ) +
                theme(panel.border=element_rect(colour='black', fill=NA))

ggplotly(box_build)

#Histograms 
#Distribution of WAP count by building and floor
hist_bf <-  ggplot(data_full, aes(x=WAP_num, fill=FLOOR)) + geom_bar() + 
# ver como hacer density line  -> stat_density(aes(group = BUILDINGID, color = "red"),position="identity",geom="line")+
                facet_grid(BUILDINGID~.) +
                theme(text = element_text(size=14)) +
                ggtitle('Distribution of Detected Wireless Access Points by Building and Floor') +
                labs(x="Number of WAP's Detected by Building", y= 'Counts by Building Floor') +
                theme(panel.border=element_rect(colour='black', fill=NA))

#### Ignacio: Incorrect variable to plot :P
#ggplotly(box_build)
ggplotly(hist_bf)

####------------------------------------------- Feature Engineering --------------------####

# #Removing unnecessary columns
# vars_to_delete <- c("SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP")
# for (c in vars_to_delete) {
#   data_full[,vars_to_delete] <- NULL
# }
# rm(vars_to_delete)
# 
# 
# # remove WAP_num because I don't need this variable anymore
# data_full$WAP_num <- NULL
# 
# 
# ##creating a column with the highest WAP and highest Signal detected.
# #### Ignacio: This is the way to get all the columns which names contains the 
# #### the word "WAP"
# WAPS<-grep("WAP", names(data_full), value=T) #value = TRUE returns the values instead of the vectors
# 
# # Create new variables HighWAP and HighRSSI
# data_full<-data_full %>%
#   mutate(HighWAP=NA)
# 
# #gets which columns has the highest value for a fingerprint
# data_full<-data_full %>% 
#   mutate(HighWAP=colnames(data_full[WAPS])[apply(data_full[WAPS],1,which.max)])
# 
# #Function to find 1st, 2nd, 3rd highest WAPs and RSSI values 
# maxn <- function(n) function(x) order(x, decreasing = TRUE)[n]
# 
# #Set 2nd highest WAP number
# data_full <- data_full %>% 
#   mutate(HighWAP2=colnames(data_full[WAPS])[apply(data_full[WAPS],1,maxn(2))])
# 
# #Set 3rd highest WAP number
# data_full<-data_full %>% 
#   mutate(HighWAP3=colnames(data_full[WAPS])[apply(data_full[WAPS],1,maxn(3))])
# 
# #Set 4th highest WAP number
# data_full<-data_full %>% 
#   mutate(HighWAP4=colnames(data_full[WAPS])[apply(data_full[WAPS],1,maxn(4))])
# 
# #convert NA's to -110
# data_full[is.na(data_full)] <- -110
# 
# #Changing timestamp variables to factor using a For
# #### Ignacio: This is ok, but you need to be cautious in doing this step as
# #### the "as.factor" function ONLY "see" the values it takes the variable
# #### enclosed within parentehses. Thus, if in a new dataset you have more 
# #### posible values than in you sampled data, you will get an error.
# vars_to_factor <- c("HighWAP", "HighWAP2", "HighWAP3", "HighWAP4")
# for (v in vars_to_factor) {
# #  data_full[,v] <- as.factor(data_full[,v])
#    data_full[,v] <- factor(data_full[,v],levels = colnames(data_full[,1:520]),
#                            labels = colnames(data_full[,1:520]))
# }
# rm(vars_to_factor)
# 
# # reorder columns 
# last_col <- c(521:530)
# data_full <-  data_full %>% select(last_col, everything())
# rm(last_col)

#saving data_full in case I have to reload the 
#saveRDS(data_full, file = "WiFi_Locationing/data/data_full.rds")
saveRDS(data_full, file = "../data/data_full.rds")
#data_full <- readRDS("WiFi_Locationing/data/data_full.rds")
data_full <- readRDS("../data/data_full.rds")

# Remove WAPS with no variance ---- I didn't use this preprocess in my model
# Select Relevant WAPS
WAPS_VarTrain<-nearZeroVar(data_full[data_full$type=="train",WAPS], saveMetrics=TRUE)
WAPS_VarTest<-nearZeroVar(data_full[data_full$type=="test",WAPS], saveMetrics=TRUE)

db_full<-data_full[-which(WAPS_VarTrain$zeroVar==TRUE | 
                              WAPS_VarTest$zeroVar==TRUE)]   # 530 -> 322 variables
rm(WAPS_VarTrain, WAPS_VarTest)
# For modeling we keep data_full df, we didn't use db_full, but save it for future.
#db_full <- saveRDS(db_full, file = "WiFi_Locationing/data/db_full.rds")
db_full <- saveRDS(db_full, file = "../data/db_full.rds")

## Create a funtiont to count WAPs with signal higher than -65 (good signal) and lower than -65 (poor signal)
signal_count <- function(v) {
  sum(mapply(function(x) ifelse(x > -65,1,0),v))
}
## Apply to repeat the funtion for each row and create a column with the number of WAPs with good signal
data_full$num_good_signal <- apply(data_full[,c(11:530)], 1, signal_count)

# Uso la funcion discretize para dividir cuales considero que tienen alta, media o baja covertura de signal
data_full$intensity <- discretize(data_full[,531], method = "fixed", breaks = c(0,3,10,38),
                                labels = c("low_coverage", "good_coverage", "high_coverage"), include.lowest = TRUE, right = FALSE, dig.lab = 3,
                                ordered_result = FALSE, infinity = FALSE, onlycuts = FALSE)


#saveRDS(data_full, file = "WiFi_Locationing/data/data_full.rds")
saveRDS(data_full, file = "../data/data_full.rds")



