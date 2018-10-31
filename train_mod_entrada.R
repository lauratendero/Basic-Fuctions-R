### PRUEBA DATABRICKS RSTUDIO funciones


# Set library paths -------------------------------------------------------

#.libPaths("/R/libraryBDL")
#.libPaths() 

# Load libraries  ---------------------------------------------------------

library(tidyverse)
library(xgboost)
library(Matrix)
library(caret)
library(dplyr)

#' Funcion modelizacion modelo entrada
#'
#'Esta funcion entrena el modelo de entrada
#'@param mes_entrenar  mes a entrenar, entre " " 
#'@param nthread  número de cores a usar para entrenar el modelo
#'@keywords train
#'@export
#'@examples train_mod_entrada("201802",20)
#'@author Laura Tendero Ossorio
#'@usage train_mod_entrada(mes_entrenar, nthread)
#'@usage Crea el directorio /entrenamiento/mes_entrenar/modelizacion, donde se guardaran todos los resultados
#'@references Banc Sabadell


# ### Parameters to the function:mes_entrenar = mes a entrenar.
#        Ejemplo: "201806"
# nthread = número de cores a usar para entrenar el modelo
#        Ejemplo: 20

train_mod_entrada <- function(mes_entrenar, nthread) {
  
#mes_entrenar <- "201802"
#nthread <- 24

# Create directory ---------------------------------------------------------------

model_path <- paste0("/dbfs/mnt/pa/funciones/", mes_entrenar)

dir.create(model_path, mode= "0777")

# Generate logs ---------------------------------------------------------------     

sink(file=paste0(model_path, "/model_log_entrenamiento_",Sys.Date(),".txt"), split= TRUE) 

time_mod <- as.character(Sys.Date())   
cat(paste(time_mod), "\n") 

# Load data ---------------------------------------------------------------

cat("loading data...", "\n")

#etl_path <- gsub("modelizacion", "etl", model_path)

#TABLA_FINAL_ENTRADA <- readRDS( file = paste0(etl_path, "/TABLA_FINAL_ENTRADA.RDS"))
#TABLA_FINAL_ENTRADA <- TABLA_FINAL_ENTRADA %>%
#filter(!(id_fch_datos_yyyymm %in% c('201701','201702','201703','201704','201705','201706','201707')))

TABLA_FINAL_ENTRADA <- readRDS(file="/dbfs/mnt/pa/DATASET_MODELO_DATABRICKS.RDS")
TABLA_FINAL_ENTRADA <- cbind(clave_encrypt=TABLA_FINAL_ENTRADA[,109], TABLA_FINAL_ENTRADA[,1:108])
# VALIDATION <- TABLA_FINAL_ENTRADA[492967:794673,]
# saveRDS(VALIDATION, file=paste0(model_path, "/validation.RDS"))
# TABLA_FINAL_ENTRADA <- TABLA_FINAL_ENTRADA[1:492966,]
saveRDS(TABLA_FINAL_ENTRADA, file=paste0(model_path, "/train_test.RDS"))

cat("The dimension of TABLA FINAL ENTRADA is (var = 111):", dim(TABLA_FINAL_ENTRADA),  "\n")

# Check dates ---------------------------------------------------------------

cat("Training period is:", min(TABLA_FINAL_ENTRADA$id_fch_datos_yyyymm),max(TABLA_FINAL_ENTRADA$id_fch_datos_yyyymm), "\n")

dataset <- TABLA_FINAL_ENTRADA

# Split data into train/test ---------------------------------------------------------------

ini.seed =  1273
set.seed(ini.seed)
inTrain  <- caret::createDataPartition(dataset$variable_objetivo , p=0.80 ,list = FALSE )
training <- dataset[ sort(inTrain),]
testing  <- dataset[-sort(inTrain),]

# Create separate vectors of our outcome variable for both our train and test sets ----------------

train_label <- as.vector(training$variable_objetivo)
test_label  <- as.vector(testing$variable_objetivo)

# Create sparse matrixes and perform One-Hot Encoding to create dummy variables -------------------

#dtrain    <- sparse.model.matrix(variable_objetivo ~ .-1, data = training[,-(1:3)])
#dtest     <- sparse.model.matrix(variable_objetivo ~ .-1, data = testing[,-(1:3)])

dtrain    <- sparse.model.matrix(variable_objetivo ~ .-1, data = training[,-1])
dtest     <- sparse.model.matrix(variable_objetivo ~ .-1, data = testing[,-1])

cat("Sparse Matrices dimensions are (172 var):", "train", dim(dtrain),"test", dim(dtest) ,  "\n")

# Sparse matrices ---------------------------------------------------------------------------------

dtrainM   <- xgboost::xgb.DMatrix(data = dtrain,label = train_label) 
dtestM    <- xgboost::xgb.DMatrix(data = dtest ,label = test_label)

# Save tables -------------------------------------------------------------------------------------

cat("saving tables...", "\n")

saveRDS(testing, file=paste0(model_path, "/testing.RDS"))
saveRDS(training, file=paste0(model_path, "/training.RDS"))

saveRDS(dtrainM, file=paste0(model_path, "/dtrainM.RDS"))
saveRDS(dtestM, file=paste0(model_path, "/dtestM.RDS"))

saveRDS(dtrain, file=paste0(model_path, "/dtrain.RDS"))
saveRDS(dtest, file=paste0(model_path, "/dtest.RDS"))

saveRDS(train_label, file=paste0(model_path, "/train_label.RDS"))
saveRDS(test_label, file=paste0(model_path, "/test_label.RDS"))


# Train data with optimal parameters ---------------------------------------------------------------

cat("Executing training modelo entrada", "\n")

nround = 9971

param <- list(booster          = "gbtree",
              objective        = "binary:logistic",
              eta              = 0.007859093,  
              gamma            = 1     ,   
              max_depth        = 16    ,
              min_child_weight = 20    ,
              subsample        = 0.8   , 
              colsample_bytree = 0.7933569   ,
              scale_pos_weight = 36    ,
              max_delta_step   = 8     )

cat("Paramater nround is:",nround,  "\n")

cat("Training parameters are:", "\n")
print(as.data.frame(param))

cat("Training model...", "\n")

time_train <- Sys.time()

seed = 774
set.seed(seed)

mod_entrada_train <- xgb.train( data = dtrainM,
                                params = param,
                                nthread = nthread,
                                nrounds = nround,
                                print_every_n = 9,
                                watchlist = list(Test=dtestM),
                                eval_metric = "error",
                                #early.stop.round = 300,
                                maximize = F,
                                seed= 774)

(end_time <- Sys.time() - time_train)

# Save model ------------------------------------------------------------------------------------

cat("saving model...", "\n")

saveRDS(mod_entrada_train, file=paste0(model_path, "/mod_entrada_train.RDS"))


# Predict test ------------------------------------------------------------------------------------

cat("Prediction test", "\n")

prediction_test <- predict(mod_entrada_train, dtestM)

# Curva ROC y calculo del punto de corte óptimo -------------------------------------------------------------
xgb.pred <- ROCR::prediction(prediction_test, test_label)
xgb.perf <- ROCR::performance(xgb.pred, "tpr", "fpr")
auc_perf <- ROCR::performance(xgb.pred, measure = "auc")

# KS-statistic:
xgb.perf@alpha.values[[1]][xgb.perf@alpha.values[[1]]==Inf]<- quantile(xgb.perf@alpha.values[[1]],0.995)
KS.matrix= cbind(abs(xgb.perf@y.values[[1]]-xgb.perf@x.values[[1]]), xgb.perf@alpha.values[[1]])
posi= sort( KS.matrix[,1] , index.return=TRUE )$ix[nrow(KS.matrix)] 
KS_stat = KS.matrix[posi,1]      #  el valor del estadístico de KS 
opt_cutpoint = KS.matrix[posi,2] # 

cat("The cutpoint is:", opt_cutpoint, "\n")

saveRDS(opt_cutpoint, file=paste0(model_path, "/opt_cutpoint.RDS"))
saveRDS(KS_stat, file=paste0(model_path, "/KS_stat.RDS"))
saveRDS(auc_perf, file=paste0(model_path, "/auc_perf.RDS"))
saveRDS(xgb.pred, file=paste0(model_path, "/xgb.pred.RDS"))


# Confusion Matrix ------------------------------------------------------------------------------------

pred.resp.test <- ifelse(prediction_test>= opt_cutpoint, 1, 0)

cat("Tasa de mora estimada:", "\n")
print(100*table(pred.resp.test)/nrow(dtestM))

cat("Tasa de mora real:", "\n")
print(100*table(testing$variable_objetivo)/nrow(testing))

##### ERROR en confusionMatrix: 
### Error: `data` and `reference` should be factors with the same levels.
pred.resp.test <- factor(pred.resp.test, levels=c('0','1'))
test_label <- factor(test_label, levels=c('0','1'))
####
conf_mat_test <- caret::confusionMatrix(pred.resp.test, test_label, positive="1")

cat("Test confusion matrix:", "\n")
print(conf_mat_test)

# Save results ------------------------------------------------------------------------------------

cat("Saving results...", "\n")

saveRDS(conf_mat_test, file=paste0(model_path, "/conf_mat_test.RDS"))
saveRDS(prediction_test, file=paste0(model_path, "/prediction_test.RDS"))
saveRDS(pred.resp.test, file=paste0(model_path, "/pred.resp.test.RDS"))

# Importance variables ------------------------------------------------------------------------------------

importance <-  xgb.importance(feature_names = colnames(dtest), model = mod_entrada_train)

saveRDS(importance, file=paste0(model_path, "/importance.RDS"))



sink()      

Sys.chmod(list.files(model_path, all.files = TRUE, full.names = TRUE, recursive = TRUE), mode="0777",use_umask = FALSE)
}


#train_mod_entrada("prueba_iteraciones2", 10)


#' Funcion confusion matrix por segmentos y meses
#'
#'Esta funcion crea la matriz de confusion del modelo de entrada por segmentos y por segmentos y meses
#'@param test_label Ruta completa donde se encuentra el dataset testing.RDS, entre " "
#'@param prediction Ruta completa donde se encuentra la prediccion pred.resp.test.RDS, entre " "
#'@keywords confusion_matrix
#'@export 
#'@examples conf_matrix_segmento_mes("/R/BDLshared/dtap_modelo_entrada_empresas/ejecuciones/201802/entrenamiento/modelizacion_GRID_20181009/Validacion/dataset_valid.RDS",
#'"/R/BDLshared/dtap_modelo_entrada_empresas/ejecuciones/201802/entrenamiento/modelizacion_GRID_20181009/Validacion/predicted_target_valid.RDS")
#'@author Laura Tendero Ossorio
#'@usage conf_matrix_segmento_mes(test_label,prediction)
#'@usage Crea la matriz de confusion por segmentos y por segmentos y meses y guarda los resultados en la misma ruta donde se encuentran las predicciones y el test
#'@references Banc Sabadell
#'@import tidyverse
#'@import xgboost


#leemos el dataset y las predicciones

# testing<-readRDS("/R/BDLshared/dtap_modelo_entrada_empresas/ejecuciones/201802/entrenamiento/modelizacion_GRID_20181009/Validacion/dataset_valid.RDS")
# pred.resp.test<-readRDS("/R/BDLshared/dtap_modelo_entrada_empresas/ejecuciones/201802/entrenamiento/modelizacion_GRID_20181009/Validacion/predicted_target_valid.RDS")


conf_matrix_segmento_mes<-function(test_label,prediction) {
  
  # Matriz de confusion por segmentos
  
  # me daba error la funcion lapply, porque tenia niveles con variable objetivo solo 0 o pred solo 0 que era caracter
  # tengo q pasarlo a factor para que me calcule la confusion matrix
  # tambien tengo un nivel que tiene 0 registros, entonces en la lista me crea un NUll y da error la confusion matrix
  # como se que es en el nivel K.F200M lo elimino directamente
  
  
  testing<-readRDS(test_label)
  pred.resp.test<-readRDS(prediction)
  
  test_segmentos <- testing%>%
    cbind( pred=pred.resp.test) %>%
    dplyr::select(me_nivel_factur_v1 , variable_objetivo, pred ) %>%
    mutate_if(is.double, as.factor) %>% 
    mutate(me_nivel_factur_v1 = as.character(me_nivel_factur_v1)) %>%
    #filter(!me_nivel_factur_v1 %in% c( "K.F200M")) %>%
    tbl_df() %>% 
    split(list(.$me_nivel_factur_v1))
  
  cm_test_segmentos <- lapply(test_segmentos, function(x) caret::confusionMatrix(x$pred, x$variable_objetivo, positive="1"))
  path<-gsub("dataset_valid.RDS","",test_label)
  #saveRDS(cm_test_segmentos,paste0(path,"/cm_test_segmentos.RDS")
  print(cm_test_segmentos)
  
  cat("Matriz de confusion por segmentos y mes")
  
  # Aqui ocurre lo mismo, pero nose que elemento me sale 0, asique paso mes a character y asi no me creara esos elementos con 0 registros en la lista
  # Me sigue creando elementos en la lista con 0 registros, tengo q eliminarlos para que no me de error
  
  testing<-readRDS(test_label)
  pred.resp.test<-readRDS(prediction)
  
  test_segmentos_mes <- testing%>%
    cbind( pred=pred.resp.test) %>%
    dplyr::select(me_nivel_factur_v1 ,mes, variable_objetivo, pred ) %>%
    mutate(variable_objetivo=factor(variable_objetivo,levels=c("0","1"))) %>% 
    mutate(pred=factor(pred,levels=c("0","1"))) %>% 
    mutate(mes = as.character(mes)) %>%
    mutate(me_nivel_factur_v1 = as.character(me_nivel_factur_v1))%>%
    tbl_df() %>% 
    split(list(.$me_nivel_factur_v1,.$mes))%>%
    .[lapply(.,nrow)>0]
  
  
  cm_test_segmentos_mes <- lapply(test_segmentos_mes, function(x) caret::confusionMatrix(x$pred, x$variable_objetivo, positive="1"))
  #saveRDS(cm_test_segmentos_mes,paste0(path,"/cm_test_segmentos.RDS")
  print(cm_test_segmentos_mes)   
}
