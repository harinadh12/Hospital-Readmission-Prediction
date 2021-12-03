library(tidyverse)
library(caret)
library(janitor)
library(Matrix)
library(xgboost)
library(dplyr)
library(MLmetrics)
library(lightgbm)
library(data.table)
install.packages('embed', dependencies= TRUE)
library(embed)
#install.packages('embed')
# Read train file
train.data <- read.csv("../input/ida-files/hm7-Train.csv",stringsAsFactors = F)

#Read test file
test.data <- read.csv("../input/ida-files/hm7-Test.csv", stringsAsFactors =F)


# combine train and test data for summarization
test.data$readmitted <- NA
full.data <- rbind(train.data, test.data)
#summary(full.data)


#Let's look at the response variable - Is it balanced?
full.data %>% tabyl(readmitted)
## -- Yes it is balanced 47% readmitted and 53% donot

#removing near zero variance predictor variables to reduce computation cost
#nzv <- nearZeroVar(full.data)
#full.data <- full.data[,-nzv]


full.data$payer_code[is.na(full.data$payer_code)] <- "UNK"
full.data$race[is.na(full.data$race)] <- "UNK"

full.data$medical_specialty[is.na(full.data$medical_specialty)] <- "UNK"
#full.data$diagnosis[is.na(full.data$diagnosis)] <- "UNK"
full.data$readmitted <- as.factor(full.data$readmitted)
full.data$diag_desc[is.na(full.data$diag_desc)] = "UNK"


full.data$admission_type <- as.factor(full.data$admission_type)
full.data$discharge_disposition <- as.factor(full.data$discharge_disposition)
full.data$admission_source <- as.factor(full.data$admission_source)



keys = c('metformin','repaglinide','nateglinide','glipizide', 'glyburide', 'pioglitazone', 
         'rosiglitazone','chlorpropamide','glimepiride','acetohexamide',
         'tolbutamide','acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton',
         'glyburide.metformin','glipizide.metformin','glimepiride.pioglitazone','metformin.rosiglitazone','metformin.pioglitazone')

full.data$med_up_cnt = 0
full.data$med_dwn_cnt = 0
full.data$med_std_cnt = 0
full.data$net_change =0

for(key in keys){
  full.data$med_up_cnt <- ifelse((full.data[key] == 'Up' ),  full.data$med_up_cnt  + 1, full.data$med_up_cnt)
  full.data$med_dwn_cnt <- ifelse((full.data[key] == 'Down'), full.data$med_dwn_cnt  + 1, full.data$med_dwn_cnt)
  full.data$med_std_cnt <- ifelse((full.data[key] == 'Steady'), full.data$med_std_cnt  + 1, full.data$med_std_cnt)
  full.data$net_change <- ifelse((full.data[key] == 'Up'), full.data$net_change  + 1, ifelse(full.data[key] == 'Down',full.data$net_change-1,full.data$net_change))
}


full.data <-  full.data  %>% select(!c('metformin','repaglinide','nateglinide','glipizide', 'glyburide', 'pioglitazone', 
                                       'rosiglitazone','chlorpropamide','glimepiride','acetohexamide',
                                       'tolbutamide','acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton',
                                       'glyburide.metformin','glipizide.metformin','glimepiride.pioglitazone',
                                       'metformin.rosiglitazone','metformin.pioglitazone'))

full.data$gender[full.data$gender=='Unknown/Invalid'] = "Unknown"
full.data$A1Cresult[full.data$A1Cresult==">7"] = "Medium"
full.data$A1Cresult[full.data$A1Cresult==">8"] = "High"
full.data$max_glu_serum[full.data$max_glu_serum==">200"] = "Medium"
full.data$max_glu_serum[full.data$max_glu_serum==">300"] = "High"

#temp <- full.data %>% mutate_if(is.character, as.factor)
train.data = full.data[!is.na(full.data$readmitted),]

hosp_recipe <- recipe(train.data) %>% 
  update_role(age, race, gender, payer_code, A1Cresult, insulin, max_glu_serum,medical_specialty,admission_source, discharge_disposition, diag_desc,admission_type, new_role = "predictor") %>%
  update_role(readmitted, new_role = "outcome") %>%
  step_embed(age, race, gender, payer_code, A1Cresult, max_glu_serum,insulin,medical_specialty,admission_source,discharge_disposition, diag_desc,admission_type, num_terms = 4, hidden_units = 16, outcome = vars(readmitted), #<<
             options = embed_control(metrics = "categorical_accuracy", epochs = 20))
trained_recipe = prep(hosp_recipe)
full.data$diagnosis = NULL


ms = unique(train.data$medical_specialty)
med_level = trained_recipe$steps[[1]]$mapping$medical_specialty$`..level`
for(m in ms){
  full.data$medical_specialty_embed_1[full.data$medical_specialty==m] = trained_recipe$steps[[1]]$mapping$medical_specialty$medical_specialty_embed_1[med_level==m]
  full.data$medical_specialty_embed_2[full.data$medical_specialty==m] = trained_recipe$steps[[1]]$mapping$medical_specialty$medical_specialty_embed_2[med_level==m]
  full.data$medical_specialty_embed_3[full.data$medical_specialty==m] = trained_recipe$steps[[1]]$mapping$medical_specialty$medical_specialty_embed_3[med_level==m]
  full.data$medical_specialty_embed_4[full.data$medical_specialty==m] = trained_recipe$steps[[1]]$mapping$medical_specialty$medical_specialty_embed_4[med_level==m]
  
}

ms = unique(train.data$diag_desc)
med_level = trained_recipe$steps[[1]]$mapping$diag_desc$`..level`
for(m in ms){
  full.data$diag_desc_embed_1[full.data$diag_desc==m] = trained_recipe$steps[[1]]$mapping$diag_desc$diag_desc_embed_1[med_level==m]
  full.data$diag_desc_embed_2[full.data$diag_desc==m] = trained_recipe$steps[[1]]$mapping$diag_desc$diag_desc_embed_2[med_level==m]
  full.data$diag_desc_embed_3[full.data$diag_desc==m] = trained_recipe$steps[[1]]$mapping$diag_desc$diag_desc_embed_3[med_level==m]
  full.data$diag_desc_embed_4[full.data$diag_desc==m] = trained_recipe$steps[[1]]$mapping$diag_desc$diag_desc_embed_4[med_level==m]
  
  
}

ms = unique(train.data$admission_type)
med_level = trained_recipe$steps[[1]]$mapping$admission_type$`..level`
for(m in ms){
  full.data$admission_type_embed_1[full.data$admission_type==m] = trained_recipe$steps[[1]]$mapping$admission_type$admission_type_embed_1[med_level==m]
  full.data$admission_type_embed_2[full.data$admission_type==m] = trained_recipe$steps[[1]]$mapping$admission_type$admission_type_embed_2[med_level==m]
  full.data$admission_type_embed_3[full.data$admission_type==m] = trained_recipe$steps[[1]]$mapping$admission_type$admission_type_embed_3[med_level==m]
  full.data$admission_type_embed_4[full.data$admission_type==m] = trained_recipe$steps[[1]]$mapping$admission_type$admission_type_embed_4[med_level==m]
  
}

ms = unique(train.data$discharge_disposition)
med_level = trained_recipe$steps[[1]]$mapping$discharge_disposition$`..level`
for(m in ms){
  full.data$discharge_disposition_embed_1[full.data$discharge_disposition==m] = trained_recipe$steps[[1]]$mapping$discharge_disposition$discharge_disposition_embed_1[med_level==m]
  full.data$discharge_disposition_embed_2[full.data$discharge_disposition==m] = trained_recipe$steps[[1]]$mapping$discharge_disposition$discharge_disposition_embed_2[med_level==m]
  full.data$discharge_disposition_embed_3[full.data$discharge_disposition==m] = trained_recipe$steps[[1]]$mapping$discharge_disposition$discharge_disposition_embed_3[med_level==m]
  full.data$discharge_disposition_embed_4[full.data$discharge_disposition==m] = trained_recipe$steps[[1]]$mapping$discharge_disposition$discharge_disposition_embed_4[med_level==m]
  
  
}

ms = unique(train.data$admission_source)
med_level = trained_recipe$steps[[1]]$mapping$admission_source$`..level`
for(m in ms){
  full.data$admission_source_embed_1[full.data$admission_source==m] = trained_recipe$steps[[1]]$mapping$admission_source$admission_source_embed_1[med_level==m]
  full.data$admission_source_embed_2[full.data$admission_source==m] = trained_recipe$steps[[1]]$mapping$admission_source$admission_source_embed_2[med_level==m]
  full.data$admission_source_embed_3[full.data$admission_source==m] = trained_recipe$steps[[1]]$mapping$admission_source$admission_source_embed_3[med_level==m]
  full.data$admission_source_embed_4[full.data$admission_source==m] = trained_recipe$steps[[1]]$mapping$admission_source$admission_source_embed_4[med_level==m]
  
}



full.data$medical_specialty = NULL

full.data$admission_source = NULL
full.data$admission_type = NULL
full.data$discharge_disposition = NULL




full.data$diag_desc_embed_1[is.na(full.data$diag_desc_embed_1)] = mean(full.data$diag_desc_embed_1, na.rm= TRUE)
full.data$diag_desc_embed_2[is.na(full.data$diag_desc_embed_1)] = mean(full.data$diag_desc_embed_2, na.rm= TRUE)
full.data$diag_desc_embed_3[is.na(full.data$diag_desc_embed_1)] = mean(full.data$diag_desc_embed_3, na.rm= TRUE)
full.data$diag_desc_embed_4[is.na(full.data$diag_desc_embed_1)] = mean(full.data$diag_desc_embed_4, na.rm= TRUE)


full.data$medical_specialty_embed_1[is.na(full.data$medical_specialty_embed_1)] = mean(full.data$medical_specialty_embed_1, na.rm= TRUE)
full.data$medical_specialty_embed_2[is.na(full.data$medical_specialty_embed_2)] = mean(full.data$medical_specialty_embed_2, na.rm= TRUE)
full.data$medical_specialty_embed_3[is.na(full.data$medical_specialty_embed_3)] = mean(full.data$medical_specialty_embed_3, na.rm= TRUE)
full.data$medical_specialty_embed_4[is.na(full.data$medical_specialty_embed_4)] = mean(full.data$medical_specialty_embed_4, na.rm= TRUE)



full.data$diag_desc= NULL


full.data = full.data %>% mutate(age=case_when(age == "[0-10)" ~ 40,
                                               age == "[10-20)" ~ 40,
                                               age == "[20-30)" ~ 40,
                                               age == "[30-40)" ~ 40,
                                               age == "[40-50)" ~ 80,
                                               age == "[50-60)" ~ 80,
                                               age == "[60-70)" ~ 80,
                                               age == "[70-80)" ~ 80,
                                               age == "[80-90)" ~ 100,
                                               age == "[90-100)" ~ 100))




train.data = full.data[!is.na(full.data$readmitted),]
test.data = full.data[is.na(full.data$readmitted),]


str(train.data)


################################# Multi-Layer Perceptron using Caret Package ################################################

# mlp_grid = expand.grid(layer1 = 10,
#                        layer2 = 10,
#                        layer3 = 10)

# mlp_fit = caret::train(readmitted~.-patientID,
#                        data=train.data, 
#                        method = "mlpML", 
#                        #preProc =  c('center', 'scale'),
#                        trControl = trainControl(method = "cv", verboseIter = TRUE, returnData = FALSE),
#                        tuneGrid = mlp_grid)


# mlp_fit

########################################## MARS Model Using Caret Package ######################################################3

# train_control <- trainControl(method = "cv", number = 5)

# # train the model on training set
# model <- train(readmitted ~ .-patientID,
#                data = train.data,
#                trControl = train_control,
#                method = "earth",
#                #family=binomial()
#               )


# model







########################### Extreme Gradient Boosting Framework - XGBoost Using Caret Package ############################
trctrl <- trainControl(method = "repeatedcv", number = 5, repeats= 3)
tune_grid <- expand.grid(nrounds = 500,
                         max_depth = 5,
                         eta = 0.05,
                         gamma = 0.01,
                         colsample_bytree = 0.5,
                         min_child_weight = 0,
                         subsample = 0.75)

xgb_fit <- train(readmitted ~.-patientID, data = train.data, method = "xgbTree",
                trControl=trctrl,
                preProc= c("center","scale"),
                tuneGrid = tune_grid,
                tuneLength = 20) 

xgb_preds <-  xgb_fit %>% predict(test.data)

results <- data.frame(patientID=test.data$patientID,readmitted= xgb_preds)
write.csv(results,"xgb_preds.csv", row.names = FALSE, quote = FALSE)



######### Neural Networks using Keras Framework #############


testing <- test.data[,2:70]


set.seed(825)
tr_inds <- sample.int(nrow(train.data), 12000)
training <- train.data[-tr_inds,2:70]
train_target <- train.data[-tr_inds,71]
val_data <- train.data[tr_inds,2:70]
val_target <- train.data[tr_inds,71]


# also don't forget to identify the target variable (we can use one for training and test)
trainingtarget <- as.numeric(as.factor(train_target))-1  # includes the dependent variable status for the training data

# create the categorical variables
trainLables <- keras::to_categorical(trainingtarget)


training = apply(training, 2, scale)
testing  =  apply(testing, 2, scale)


val_data = apply(val_data, 2 ,scale)


val_target <- as.numeric(as.factor(val_target))-1
valLables=keras::to_categorical(val_target)


dim(training)



# create the first model design
model <- keras_model_sequential()
# the keras_model_sequential consists of a linear stack of layers (in some sequential linear order)

# now we use the pipe function (%>%) to pass info from left to right, i.e., add additonal functions to 'model'
model %>%
  layer_dense(units=120, activation = 'relu', input_shape = 69) %>%     # this is for independent variables
  #layer_dropout(rate = 0.2)%>%
  layer_dense(units = 60,activation = "relu",name = "den2")%>%
  #layer_gaussian_dropout(rate = 0.2)%>%
  layer_dense(units = 30,activation = "relu",name = "den3")%>%
  layer_dropout(rate = 0.1)%>%
  layer_dense(units=2, activation = 'sigmoid')

# ########################## Configure the model for the learning process ############################
model %>% compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics = c("accuracy")
)

# binary_crossentrophy  is used when we have categorical variables (2 options here; status)
# adam is a commonly used optimiser
# accuracy is how accurate the predicted model matches the observed result. This is the metric

history1 <- model %>%
  fit(
    x = as.matrix(training), # this is the input, the first 21 independent variables
    y = as.matrix(trainLables),
    epochs = 100,
    batch_size = 50,
    #validation_split = 0.30,
      validation_data = list(as.matrix(val_data),as.matrix(valLables)),
   # callbacks = list(
#   callback_model_checkpoint("checkpoints.h5"),
#callback_early_stopping(monitor = "val_accuracy", mode='max', min_delta=1))

)


history1


yhat_keras_class_vec <- predict_classes(object = model, x = as.matrix(testing)) %>%
    as.vector()

yhat_keras_prob_vec <- predict_proba(object = model, x = as.matrix(testing)) %>%
    as.vector()




results <- data.frame(patientID=test.data$patientID,readmitted= yhat_keras_class_vec)
write.csv(results,"mlp_keras_droput.csv", row.names = FALSE, quote = FALSE)

