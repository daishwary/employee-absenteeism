rm(list=ls())
setwd("C:/Users/asus/Desktop/project3")
#load library
library(mlr)
library(xlsx)
library(plyr)
library(corrplot)
library(ggplot2)
library(gridExtra)
library(ggthemes)
library(caret)
library(MASS)
library(party)
library(DMwR)
library(corrgram)
library(randomForest)
library(xgboost)
library(readr)
library(car)
library(stringr)
library(Metrics)
#Load comments/text
data = read.xlsx("data.xls", sheetIndex = 1,)

str(data)
#missing value analysis
sapply(data, function(x) sum(is.na(x)))

##################################Missing Values Analysis###############################################
missing_val = data.frame(apply(data,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(data)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
write.csv(missing_val, "Miising_perc.csv", row.names = F)

ggplot(data = missing_val[1:3,], aes(x=reorder(Columns, -Missing_percentage),y = Missing_percentage))+
   geom_bar(stat = "identity",fill = "grey")+xlab("Parameter")+
   ggtitle("Missing data percentage (data)") + theme_bw()

#Mean Method
#data$custAge[is.na(data$custAge)] = mean(data$custAge, na.rm = T)

#Median Method
#data$custAge[is.na(data$custAge)] = median(data$custAge, na.rm = T)

# kNN Imputation
data = knnImputation(data, k = 9)
sum(is.na(data))

write.xlsx(data, 'data_missing.xls', row.names = F)

library(rpart)
library(MASS)

#histograms
hist(data$ID)
hist(data$Reason.for.absence)
hist((data$Month.of.absence))
hist(data$Seasons)
hist(data$Distance.from.Residence.to.Work)
hist(data$Service.time)
hist(data$Work.load.Average.day.)
hist(data$Body.mass.index)

############################################Outlier Analysis#############################################
# ## BoxPlots - Distribution and Outlier Check

df = subset(data , 
                  select = -c(Seasons,Disciplinary.failure,Education,Social.drinker,Social.smoker))
cnames = colnames(df)

# 
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Absenteeism.time.in.hours"), data = subset(df))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="Absenteeism.time.in.hours")+
           ggtitle(paste("Box plot of Absenteeism.time.in.hoursed for",cnames[i])))
}

# ## Plotting plots together
gridExtra::grid.arrange(gn1,gn5,gn2,ncol=3)
gridExtra::grid.arrange(gn6,gn7,ncol=2)
gridExtra::grid.arrange(gn8,gn9,ncol=2)
gridExtra::grid.arrange(gn3,gn4,gn10,ncol=3)
gridExtra::grid.arrange(gn11,gn12,ncol=2)
gridExtra::grid.arrange(gn13,gn14,ncol=2)
gridExtra::grid.arrange(gn15,gn16,ncol=2)


# #Replace all outliers with NA and impute
# #create NA 
for(i in cnames){
  val = data[,i][data[,i] %in% boxplot.stats(data[,i])$out]
  #print(length(val))
  data[,i][data[,i] %in% val] = NA
}
# 
sum(is.na(data))
data = knnImputation(data,k=3)

#

## Correlation Plot 
corrgram(data, order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")  

#Divide the data into data and test
#set.seed(123)
data_index = sample(1:nrow(data), 0.8 * nrow(data))
train = data[data_index,]
test = data[-data_index,]

# ##rpart for regression
fit = rpart(Absenteeism.time.in.hours ~ ., data = train, method = "anova")

#Predict for new test cases
predictions_DT = predict(fit, test[,-21])

#MAPE
#calculate MAPE
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

MAPE(test[,21], predictions_DT)
rmse(test$Absenteeism.time.in.hours,predictions_DT)

#Error Rate: 10.33
#Accuracy: 89.67

#Linear Regression
#check multicollearity
library(usdm)
vif(data[,-21])

vifcor(data[,-21], th = 0.9)

#run regression model
lm_model = lm(Absenteeism.time.in.hours ~., data = data)

#Summary of the model
summary(lm_model)

#Predict
predictions_LR = predict(lm_model, test[,1:20])

#Calculate MAPE
MAPE(test[,21], predictions_LR)
rmse(test$Absenteeism.time.in.hours,predictions_LR)

#Error Rate: 8.8
#acuracy: 91.2%

###Random Forest
RF_model = randomForest(Absenteeism.time.in.hours ~ ., train, importance = TRUE, ntree = 500)

#Extract rules fromn random forest
#transform rf object to an inTrees' format
library(RRF)
library(inTrees)
treeList = RF2List(RF_model) 

# 
# #Extract rules
exec = extractRules(treeList, train[,-21])  # R-executable conditions
# 
# #Visualize some rules
exec[1:2,]
# 
# #Make rules more readable:
readableRules = presentRules(exec, colnames(train))
readableRules[1:2,]
# 
# #Get rule metrics
ruleMetric = getRuleMetric(exec, train[,-21], train$Absenteeism.time.in.hours)  # get rule metrics
# 
# #evaulate few rules
ruleMetric[1:2,]

#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test[,-21])

##Evaluate the performance of regression model
#Calculate MAPE
MAPE(test[,21], RF_Predictions)
rmse(test$Absenteeism.time.in.hours,RF_Predictions)

#Accuracy = 93.4%
#Error rate = 6.6%


#XGBoost


#first default - model training
xgbFit=xgboost(data=as.matrix(train[,-21]),nfold=5,label=as.matrix(train$Absenteeism.time.in.hours),nrounds=2200,verbose=FALSE,objective='reg:linear',eval_metric='rmse',nthread=8,eta=0.01,gamma=0.0468,max_depth=6,min_child_weight=1.7817,subsample=0.5213,colsample_bytree=0.4603)

#model prediction
xgbPredict = predict(xgbFit, newdata = as.matrix(test[, -21]))
rmse(test$Absenteeism.time.in.hours, xgbPredict)

#Calculate Error rate
MAPE(test[,21], xgbPredict)

#Accuracy = 97.02%
#Error rate = 2.98%

#view variable importance plot
mat <- xgb.importance (feature_names = colnames(train),model = xgbFit)
xgb.plot.importance (importance_matrix = mat[1:20]) 

