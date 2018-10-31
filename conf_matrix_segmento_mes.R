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


#leemos el dataset y las predicciones

# testing<-readRDS("/R/BDLshared/dtap_modelo_entrada_empresas/ejecuciones/201802/entrenamiento/modelizacion_GRID_20181009/Validacion/dataset_valid.RDS")
# pred.resp.test<-readRDS("/R/BDLshared/dtap_modelo_entrada_empresas/ejecuciones/201802/entrenamiento/modelizacion_GRID_20181009/Validacion/predicted_target_valid.RDS")


.libPaths("/R/libraryBDL")
.libPaths()

library(psych)
library(caret)
library(Matrix)
library(xgboost)
library(tidyverse)

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
