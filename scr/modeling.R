#Load required libraries 
pacman:: p_load("rstudioapi", "readr","dplyr", "tidyr", "ggplot2", "plotly", 
                "data.table", "reshape2", "caret", "randomForest", "bbplot", 
                "doMC", "bbplot", "parallel", "iterators", "ranger", "tidyverse", "class",
                "doParallel", "arules")

data_full <- readRDS("WiFi_Locationing/data/data_full.rds")

#------ Split Data before modeling -----
data_full_split<-split(data_full, data_full$type) #split by type
list2env(data_full_split, envir=.GlobalEnv) #sent train and test df to global enviroment
rm(data_full_split) #remove data_full_split

db_train <- train
db_test <- test

####-------------- Preparing for modeling -----------------------------------------------------####

#-Assign values in seeds agrument of trainControl
set.seed(123)
# Prepare Parallel Process
getDoParWorkers() #Firstly check 
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
stopCluster(cluster)

#-Define parameters in trainControl
fitControl <- trainControl(method = "repeatedcv", # Cross Validation
                           number = 10, # 10 fold
                           repeats=3,
                           allowParallel = TRUE)
####-------------------------------------------------- Models 
####---------------------------------------------------------------------- BUILDING predictions -----------####
#------------------------------------------------------------------------------/ SVM Algorithm
svm_B1 <- caret::train(BUILDINGID~HighWAP + HighWAP2 + HighWAP3, data=db_train,
                        method='svmLinear',
                        trControl = fitControl)
# saveRDS(svm_B_1, "../WiFi_Locationing/models/svm_B_1.rds")
svm_B1 <- readRDS("WiFi_Locationing/models/svm_B1.rds")
# test
predic_B_1_test <- predict(svm_B1, db_test)
B1_cMatrix_svm <- confusionMatrix(predic_B_1_test, db_test$BUILDINGID)
#   Accuracy   Kappa    
#   0.9991    0.9986

#------------------------------------------------------------------------------//   KNN Algorithm
# test
system.time(kNN_B2 <- knn(db_train[,11:530], db_test[,11:530], cl =  db_train$BUILDINGID, k = 5))
# 
# saveRDS(kNN_B1_test, "models/kNN_B1_test.rds")
kNN_B2 <- readRDS("WiFi_Locationing/models/kNN_B2.rds")
B2_cMatrix_knn <- confusionMatrix(kNN_B2, db_test$BUILDINGID)
#   Accuracy   Kappa    
#   0.9973    0.9957

####---------------------------------------------------------------------- FLOOR predictions -----------####
###---------------------------------------------------------------------------/  SVM Algorithm

F1_SVM <- caret::train(FLOOR~HighWAP + HighWAP2 + HighWAP3, data=db_train,
                        method='svmLinear',
                        trControl = fitControl)
saveRDS(F1_SVM, "WiFi_Locationing/models/svm_F1.rds")
svm_F1<- readRDS("WiFi_Locationing/models/svm_F1.rds")

# test
F1_SVM_pred_test <- predict(svm_F1, db_test)
F1_cMatrix_svm <- confusionMatrix(F1_SVM_pred_test, db_test$FLOOR)
#   Accuracy   Kappa    
#   0.9118    0.8766

###---------------------------------------------------------------------------//  Random Forest Algorithm

# Getting the optimal value of mtry with tuneRF funtion
bestmtry_rf<-tuneRF(db_train[,12:530], db_train$FLOOR, ntreeTry=100,
                    stepFactor=2,improve=0.05,trace=TRUE, plot=T) 

system.time(rf_F1<-randomForest(y=db_train$FLOOR,x=db_train[,12:530],importance=T,
                                method="rf", ntree=100, mtry=44))

# saveRDS(rf_F1, "models/rf_F1.rds")
rf_F1 <- readRDS("WiFi_Locationing/models/rf_F1.rds")
##------------------------------------ test
rf_pred_F1_test <- predict(rf_F1, db_test)
F2_cMatrix_rf <- confusionMatrix(rf_pred_F1_test, db_test$FLOOR)
#   Accuracy   Kappa 
#   0.9145     0.8803
# Best model because it has 78 errors in cMatrix and svm has 88

####---------------------------------------------------------------------- LONGITUD predictions -----------####
###---------------------------------------------------------------------------/  KNN 
system.time(knn_long_1 <- knnreg(LONGITUDE~., data = db_train[,c(1,11:530)]))

saveRDS(knn_long_1, "../WiFi_Locationing/models/knn_long_1.rds")
knn_long_1 <- readRDS("WiFi_Locationing/models/knn_long_1.rds")

pred_long_1_test <- predict(knn_long_1, db_test)
postRsmpl_long_1_test <- postResample(pred_long_1_test, db_test$LONGITUDE)
# RMSE         Rsquared        MAE 
# 11.6794071   0.9905899       6.0801572 
error_LON_knn<- db_test$LONGITUDE - pred_long_1_test


####--------------------------------------------------------------------------//  RF Model 
# Saving the waps in a vector, excluding HighWAPs variables
WAPs<-grep("WAP", names(db_train[,-c(7:10)]), value=T)

# Get the best mtry value using tuneRF funtion
bestmtry_rf_1 <- tuneRF(db_train[WAPs], db_train$LONGITUDE, ntreeTry=100,
                        stepFactor=2,improve=0.05,trace=TRUE, plot=T) 

# Train a random forest using that mtry
system.time(rf_LON_1 <-randomForest(y=db_train$LONGITUDE,x=db_train[WAPs],importance=T,
                                    method="rf", ntree=100, mtry=87))

saveRDS(rf_LON_1, "WiFi_Locationing/models/rf_LON_1.rds")
rf_LON_1 <- readRDS("WiFi_Locationing/models/rf_LON_1.rds")
##------------------------------------ test
rf_pred_LON_1_test <- predict(rf_LON_1, db_test)
postRsmpl_rf_pred_LON_1_test <- postResample(rf_pred_LON_1_test, db_test$LONGITUDE)
# RMSE   Rsquared        MAE 
# 10.8664012  0.9922207  7.1546304 
error_LON_rf<- db_test$LONGITUDE - rf_pred_LON_1_test

####---------------------------------------------------------------------- LATITUD predictions -----------####
###---------------------------------------------------------------------------/  KNN - Caret

#----- Grid of k values to search
# knn_grid <- expand.grid(.k=c(1:5))

# knn_LAT_1_caret <- system.time(train(LATITUDE~., data=db_train[,c(2,11:530)], method='knn',  # NOT_RUN
#                              tuneGrid=knn_grid,trControl = fitControl))
# Caret demora mucho tiempo

system.time(knn_LAT_1 <- knnreg(LATITUDE~., data = db_train[,c(2,11:530)]))

saveRDS(knn_LAT_1, "WiFi_Locationing/models/knn_LAT_1.rds")
knn_LAT_1 <- readRDS("WiFi_Locationing/models/knn_LAT_1.rds")

# kNN_FL_tr <- readRDS("../WiFi_Locationing/models/kNN_FL_tr.rds")
predic_LAT_1_test <- predict(knn_LAT_1, db_test)
postRsmpl_LAT_1_test <- postResample(predic_LAT_1_test, db_test$LATITUDE)
# RMSE         Rsquared        MAE 
# 10.3689961   0.9785834     5.6534032 
error_LAT_knn<- db_test$LATITUDE- predic_LAT_1_test

####--------------------------------------------------------------------------//  RF Model 
# Get the best mtry value using tuneRF funtion
bestmtry_rf_LAT_2 <- tuneRF(db_train[WAPs], db_train$LATITUDE, ntreeTry=100,
                            stepFactor=2,improve=0.05,trace=TRUE, plot=T) 

# Train a random forest using that mtry
system.time(rf_LAT_2 <-randomForest(y=db_train$LATITUDE,x=db_train[WAPs],importance=T,
                                    method="rf", ntree=100, mtry=173))

saveRDS(rf_LAT_2, "WiFi_Locationing/models/rf_LAT_2.rds")
rf_LAT_2 <- readRDS("WiFi_Locationing/models/rf_LAT_2.rds")
##------------------------------------ test
rf_pred_LAT_2_test <- predict(rf_LAT_2, db_test)
postRsmpl_rf_pred_LAT_2_test <- postResample(rf_pred_LAT_2_test, db_test$LATITUDE)
# RMSE   Rsquared        MAE 
# 10.4242  0.9791802  6.5463941 
error_LAT_rf<- db_test$LATITUDE- rf_pred_LAT_2_test

####----------------------------------------------- Analyzing Erros of Models ---------------####
#### BUILDING ####
#Creating a new  data Set for Predictions and add a column for each prediction
db_predic <- as.data.frame(db_test[,-c(5:530)])  
db_predic$pred_B1_svm <- predic_B_1_test
db_predic$err_B1_svm <- ifelse(db_predic$BUILDINGID == db_predic$pred_B1_svm, "1", "0")
db_predic$pred_B2_knn <- kNN_B2
db_predic$err_B2_knn <- ifelse(db_predic$BUILDINGID == db_predic$pred_B2_knn, "1", "0")

# Use Paste function to create an ID for Compare errors in diferent BUildings Prediction Models 
db_predic <-  within(db_predic, errB1_B2 <- paste(err_B1_svm, err_B2_knn, sep = ''))

db_predic[which(db_predic$errB1_B2 == "01"),]$errB1_B2 <- "SVM Errors"
db_predic[which(db_predic$errB1_B2 == "11"),]$errB1_B2 <- "SVM / K-NN ok"
db_predic[which(db_predic$errB1_B2 == "10"),]$errB1_B2 <- "K-NN Errors"

plot_ly(db_predic) %>%
  add_markers(x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR,color = ~errB1_B2, colors = c('red', 'blue','green'),
              mode = "line", marker = list(size = 5)) %>%
  layout(title = "Building ID Predicions and Errors with Random Forest and K-NN")

#### FLOOR ####
# add a column for each Floor prediction
db_predic$pred_F1_rf <- rf_pred_F1_test
db_predic$err_F1_rf <- ifelse(db_predic$FLOOR == db_predic$pred_F1_rf, "1", "0")
db_predic$pred_F1_svm <- F1_SVM_pred_test
db_predic$err_F1_svm <- ifelse(db_predic$FLOOR == db_predic$pred_F1_svm, "1", "0")

# Use Paste function to create an ID for Compare errors in diferent Floor Prediction Models 
db_predic <- within(db_predic, errF1_F2 <- paste(err_F1_rf, err_F1_svm, sep = ''))

db_predic[which(db_predic$errF1_F2 == "00"),]$errF1_F2 <- 'RF and SVM Errors'
db_predic[which(db_predic$errF1_F2 == "11"),]$errF1_F2 <- 'Good Predictions'
db_predic[which(db_predic$errF1_F2 == "10"),]$errF1_F2 <- 'SVM Errors'
db_predic[which(db_predic$errF1_F2== "01"),]$errF1_F2 <- 'RF Errors'

# saveRDS(db_predic, file = "WiFi_Locationing/data/db_predic.rds")
# db_predic <- readRDS("data/db_predic.rds")

#Plots
  plot_ly(db_predic) %>%
    add_markers(x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, color = ~errF1_F2, colors = c('green','red', 'magenta','blue'),
              mode = "line", marker = list(size = 5)) %>%
    layout(title = "Floor Predicions with Random Forest and Support Vector Machine")

  plot_ly(db_predic) %>%
    add_markers(x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR,color = ~intensity, colors = c('red', 'blue','green'),
                mode = "line", marker = list(size = 5)) %>%
    layout(title = "WAPs with different Signal Coverage")

plot_ly(db_predic) %>%
  add_markers(x = ~LONGITUDE, y = ~LATITUDE, z = ~pred_F1_svm,
              mode = "line", marker = list(size = 5, color = "blue"), name = "SVM") %>%
  add_markers(x = ~LONGITUDE, y = ~LATITUDE, z = ~pred_F1_rf,
              mode = "line", marker = list(size = 5, color = "red"), name = "RF") %>%
  add_markers(x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR,
              mode = "line", marker = list(size = 5, color = "green"), name = "Real") %>%
    layout(title = "FLoor Real vs Random Forest and Support Vector Machine")

#### LONGITUD AND LATITUDE ####
# Plot the error distribution in LONGITUDE
error_LON<-cbind(error_LON_knn, error_LON_rf)
error_LON<-melt(error_LON) %>% select(-Var1)
colnames(error_LON)<-c("Model", "Value")

p_err_LON <- ggplot(error_LON, aes(x=Value, fill=Model)) + 
               geom_histogram(alpha = 0.4, aes(y = ..count..), position = 'identity') +
               scale_fill_brewer(palette = "Dark2") +
               scale_x_continuous(breaks=seq(-100, 100, 10)) +
               scale_y_continuous(breaks=seq(0, 700, 100)) 

ggplotly(p_err_LON)

# Plot the error distribution in LATITUDE predictions
error_LAT<-cbind(error_LAT_knn, error_LAT_rf)
error_LAT<-melt(error_LAT) %>% select(-Var1)
colnames(error_LAT)<-c("Model", "Value")

p_err_LAT <- ggplot(error_LAT, aes(x=Value, fill=Model)) + 
               geom_histogram(alpha = 0.3, aes(y = ..count..), position = 'identity') +
               labs(title="Error distribution in Latitude predictions") + 
               scale_fill_brewer(palette = "Dark2") +
               scale_x_continuous(breaks=seq(-100, 100, 10)) +
               scale_y_continuous(breaks=seq(0, 700, 100)) 

ggplotly(p_err_LAT)

## Tanto para LON y LAT, discretize los metros de errores, y contar cuantos estan en menos de 5 mts de error, entre 5 y 10,
## entre 10 y 20, + 20

error_LON$error_disc <- discretize(error_LON$Value, method = "fixed", breaks = c(-150,-20,-10,-5, 5, 10, 20, 170),
                                   labels = c("< 20", "10-20", "5-10", ">5", "5-10","10-20", "< 20"), include.lowest = TRUE, 
                                   right = FALSE, dig.lab = 3, ordered_result = FALSE, infinity = FALSE, onlycuts = FALSE)

error_LAT$error_disc <- discretize(error_LAT$Value, method = "fixed", breaks = c(-150,-20,-10,-5, 5, 10, 20, 170),
                                   labels = c("< 20", "10-20", "5-10", ">5", "5-10","10-20", "< 20"), include.lowest = TRUE, 
                                   right = FALSE, dig.lab = 3, ordered_result = FALSE, infinity = FALSE, onlycuts = FALSE)

error_LON_tbl <- as_tibble(error_LON)
  # LON_kNN_errLess5mts <- dplyr::filter(error_LON_tbl, error_disc == ">5", Model == "error_LON_knn") 
  # LON_rf_errLess5mts <- dplyr::filter(error_LON_tbl, error_disc == ">5", Model == "error_LON_rf")
  # LON_kNN_errBtw5_10mts <- dplyr::filter(error_LON_tbl, error_disc == "5-10", Model == "error_LON_knn") 
  # LON_rf_errBtw5_10mts <- dplyr::filter(error_LON_tbl, error_disc == "5-10", Model == "error_LON_rf")
  # LON_kNN_errBtw10_20mts <- dplyr::filter(error_LON_tbl, error_disc == "10-20", Model == "error_LON_knn") 
  # LON_rf_errBtw10_20mts <- dplyr::filter(error_LON_tbl, error_disc == "10-20", Model == "error_LON_rf")
  # 
error_LAT_tbl <- as_tibble(error_LAT)
  # LAT_kNN_errLess5mts <- dplyr::filter(error_LAT_tbl, error_disc == ">5", Model == "error_LAT_knn") 
  # LAT_rf_errLess5mts <- dplyr::filter(error_LAT_tbl, error_disc == ">5", Model == "error_LAT_rf")
  # LAT_kNN_errBtw5_10mts <- dplyr::filter(error_LAT_tbl, error_disc == "5-10", Model == "error_LAT_knn") 
  # LAT_rf_errBtw5_10mts <- dplyr::filter(error_LAT_tbl, error_disc == "5-10", Model == "error_LAT_rf")
  # LAT_kNN_errBtw10_20mts <- dplyr::filter(error_LAT_tbl, error_disc == "10-20", Model == "error_LAT_knn") 
  # LAT_rf_errBtw10_20mts <- dplyr::filter(error_LAT_tbl, error_disc == "10-20", Model == "error_LAT_rf")

## Plotting distribution of less 10 meters Errors in LONGITUDE predictions  
LON_errLess10mts <- dplyr::filter(error_LON_tbl, error_disc == ">5" | error_disc == "5-10")
p_LON_errLess10mts <- ggplot(LON_errLess10mts, aes(x=Value, fill=Model)) + 
                      geom_histogram(alpha = 0.3, aes(y = ..density..), position = 'identity') +
                      labs(title="Errors distribution of less 10 meters in Longitude predictions") + 
                      scale_fill_brewer(palette = "Dark2") +
                      scale_x_continuous(breaks=seq(-100, 100, 10)) 

ggplotly(p_LON_errLess10mts)
# k-NN best model for Longitude

## Plotting distribution of less 10 meters Errors in LATITUDE predictions  
LAT_errLess10mts <- dplyr::filter(error_LAT_tbl, error_disc == ">5" | error_disc == "5-10")
p_LAT_errLess10mts <- ggplot(LAT_errLess10mts, aes(x=Value, fill=Model)) + 
                      geom_histogram(alpha = 0.3, aes(y = ..density..), position = 'identity') +
                      labs(title="Errors distribution of less 10 meters in Latitude predictions") + 
                      scale_fill_brewer(palette = "Dark2") +
                      scale_x_continuous(breaks=seq(-100, 100, 10)) 

ggplotly(p_LAT_errLess10mts) 
# k-NN best model for Latitude

#calculate Error Total
err_total <- sqrt((error_LON_knn)^2+(error_LAT_knn)^2)
err_total <- as.data.frame(err_total)
colnames(err_total)<-("Value")

p.err_total <- ggplot(err_total, aes(x=Value)) + 
                 geom_histogram(alpha = 0.5, aes(y = ..count..), position = 'identity', fill = "#3c9bba") +
                 labs(title="Distribution of Euclidean Distance Errors") + 
                 scale_fill_brewer(palette = "Dark2") +
                 scale_x_continuous(breaks=seq(0, 200, 5)) 

ggplotly(p.err_total)  

# Calculate mean and median of Distance Error (Euclidean) 
mean(abs(err_total$Value))   # mean 9.13
median(abs(err_total$Value)) # median 5.89

# Calculate mean and median of LONGITUDE Residuals, use a funtion abs() to convert values negatives tu absolut
#K-NN 
mean(abs(error_LON_knn)) # mean 6.08
median(abs(error_LON_knn)) #median 3.45
# RF
mean(abs(error_LON_rf)) # mean 7.15
median(abs(error_LON_rf)) #median 4.79

# Calculate mean and median of LATITUDE Residuals, use a funtion abs() to convert values negatives tu absolut
#K-NN
mean(abs(error_LAT_knn)) # mean 5.65
median(abs(error_LAT_knn)) #median 3.07
# RF
mean(abs(error_LAT_rf)) # mean 6.55
median(abs(error_LAT_rf)) #median 3.99


