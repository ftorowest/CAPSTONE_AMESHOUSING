library(readr)
library(dplyr)
library(fredr)
library(lubridate)
library(AmesHousing)
library(ranger)
library(corrplot)
set.seed(123)

ames <- make_ames()

View(ames)
# informacion de que hace make_ames()
?make_ames

fredr_set_key("2ffe65efd6212331a9bffa3c1ce33caa")

# --- Crear fecha de venta ---
ames <- ames %>%
  mutate(
    Fecha_Venta = as.Date(sprintf("%d-%02d-01", Year_Sold, Mo_Sold))
  )

# --- Descargar CPI mensual desde 2006 ---
# (asegúrate de haber corrido antes: fredr_set_key("TU_API_KEY"))
cpi <- fredr(
  series_id = "CPIAUCNS",
  observation_start = as.Date("2006-01-01")
) %>%
  transmute(
    Fecha_Venta = date,   # primer día de cada mes
    CPI = value
  )

# --- Calcular CPI actual ---
cpi_actual <- cpi %>% slice_tail(n = 1) %>% pull(CPI)

# --- Unir CPI al dataset y ajustar precios ---
ames <- ames %>%
  left_join(cpi, by = "Fecha_Venta") %>%
  mutate(
    Sale_Price = Sale_Price * (cpi_actual / CPI),
    Sale_Price_Log = log(Sale_Price)
  )

View(ames)

# Eliminamos Mo_Sold, Year_Sold, Fecha_Venta y CPI (auxiliares)
ames <- ames %>% select(-c(Mo_Sold, Year_Sold, Fecha_Venta, CPI))


# Ver correlacion de Sale_Price con numericas
num <- ames[sapply(ames, is.numeric)]

M <- cor(num, use = "pairwise.complete.obs", method = "pearson")
corrplot(
  M,
  method = "color",
  type = "lower",           # solo triángulo inferior
  order = "hclust",         # agrupa por similitud
  tl.col = "black",
  tl.cex = 0.7,
  cl.cex = 0.8,
  addgrid.col = "white",
  col = colorRampPalette(c("blue", "white", "red"))(200)
)

target <- "Sale_Price"
ord <- sort(abs(M[, target]), decreasing = TRUE)
vars <- names(ord)[2:11]    # 10 primeras (excluyendo SalePrice mismo)

S <- M[c(vars, target), c(vars, target)]

corrplot(S, method = "color", type = "full", 
         tl.col = "black", tl.cex = 0.9,
         col = colorRampPalette(c("blue", "white", "red"))(200),
         addCoef.col = "black", number.cex = 0.7, cl.cex = 0.8)



# Correlaciones altas entre variables.
num <- ames[sapply(ames, is.numeric)]

num_no_price <- subset(num, select = -Sale_Price)

M <- cor(num_no_price, use = "pairwise.complete.obs", method = "pearson")

library(reshape2)
cor_long <- melt(M)

cor_long <- cor_long[cor_long$Var1 != cor_long$Var2, ]

cor_long <- cor_long[order(-abs(cor_long$value)), ]

cor_long_unique <- cor_long[!duplicated(apply(cor_long[,1:2], 1, 
                                              function(x) paste(sort(x), collapse="_"))), ]

head(cor_long_unique, 10)

#library(gridExtra)
#library(ggplot2)

#tabla_grob <- tableGrob(head(cor_long_unique, 10))

#ggsave("No_SP_Corr.png", tabla_grob, width = 8, height = 6)



# Eliminar Garage_Area de la db (ames)
ames <- ames %>% select(-Garage_Area)


# Ver regresion lineal entre Gr_Liv_Area y Fisrt_Flr_SF, Second_Flr_SF, Low_Qual_Fin_SF
modelo<- lm(Gr_Liv_Area ~ First_Flr_SF + Second_Flr_SF + Low_Qual_Fin_SF, data = ames)

# Ver resumen completo
summary(modelo)

# Prescindir de Gr_Liv_Area
ames <- ames %>% select(-Gr_Liv_Area)


# Veamos los valores NA
View(ames["Pool_QC"])
View(ames["Misc_Feature"])
View(ames["Alley"])
View(ames["Fence"])
View(ames["Fireplace_Qu"])


# Eliminamos estas columnas del análisis
ames <- ames %>% select(-c(Pool_QC, Misc_Feature, Alley, Fence, Fireplace_Qu))



# Para variables ordinales, convertimos a factor.
fac_cols <- names(ames)[sapply(ames, is.factor)]
fac_cols
qual_cols <- fac_cols[has_quality]
qual_cols
#[1] "Overall_Qual" "Overall_Cond" "Exter_Qual"   "Exter_Cond"   "Bsmt_Qual"   
#[6] "Bsmt_Cond"    "Heating_QC"   "Kitchen_Qual" "Garage_Qual"  "Garage_Cond" 

# Vamos viendo uno por uno como tratarlos

# Overall_Qual
lvl_overall <- c("Very_Poor","Poor","Fair","Below_Average","Average",
                 "Above_Average","Good","Very_Good","Excellent","Very_Excellent")

ames$Overall_Qual <- factor(ames$Overall_Qual, levels = lvl_overall, ordered = TRUE)
ames$Overall_Qual <- as.integer(ames$Overall_Qual)

# Overall_Cond
lvl_overall_cond <- c("Very_Poor","Poor","Fair","Below_Average","Average",
                      "Above_Average","Good","Very_Good","Excellent","Very_Excellent")

ames$Overall_Cond <- factor(ames$Overall_Cond, levels = lvl_overall_cond, ordered = TRUE)
ames$Overall_Cond <- as.integer(ames$Overall_Cond)

# Exter_Qual
lvl_exter <- c("Fair","Typical","Good","Excellent")

ames$Exter_Qual <- factor(ames$Exter_Qual, levels = lvl_exter, ordered = TRUE)
ames$Exter_Qual <- as.integer(ames$Exter_Qual)

# Exter_Cond
lvl_exter_cond <- c("Poor","Fair","Typical","Good","Excellent")

ames$Exter_Cond <- factor(ames$Exter_Cond, levels = lvl_exter_cond, ordered = TRUE)
ames$Exter_Cond <- as.integer(ames$Exter_Cond)

# Bsmt_Qual
lvl_bsmt <- c("No_Basement","Poor","Fair","Typical","Good","Excellent")

ames$Bsmt_Qual <- factor(ames$Bsmt_Qual, levels = lvl_bsmt, ordered = TRUE)
ames$Bsmt_Qual <- as.integer(ames$Bsmt_Qual)

# Bsmt_Cond
lvl_bsmt_cond <- c("No_Basement","Poor","Fair","Typical","Good","Excellent")

ames$Bsmt_Cond <- factor(ames$Bsmt_Cond, levels = lvl_bsmt_cond, ordered = TRUE)
ames$Bsmt_Cond <- as.integer(ames$Bsmt_Cond)

# Heating_QC
lvl_heating_qc <- c("Poor","Fair","Typical","Good","Excellent")

ames$Heating_QC <- factor(ames$Heating_QC, levels = lvl_heating_qc, ordered = TRUE)
ames$Heating_QC <- as.integer(ames$Heating_QC)

# Kitchen_Qual
lvl_kitchen <- c("Poor","Fair","Typical","Good","Excellent")

ames$Kitchen_Qual <- factor(ames$Kitchen_Qual, levels = lvl_kitchen, ordered = TRUE)
ames$Kitchen_Qual <- as.integer(ames$Kitchen_Qual)

# Garage_Qual
lvl_garage_qual <- c("No_Garage","Poor","Fair","Typical","Good","Excellent")

ames$Garage_Qual <- factor(ames$Garage_Qual, levels = lvl_garage_qual, ordered = TRUE)
ames$Garage_Qual <- as.integer(ames$Garage_Qual)

# Garage_Cond
lvl_garage_cond <- c("No_Garage","Poor","Fair","Typical","Good","Excellent")

ames$Garage_Cond <- factor(ames$Garage_Cond, levels = lvl_garage_cond, ordered = TRUE)
ames$Garage_Cond <- as.integer(ames$Garage_Cond)

# Functional
lvl_functional <- c("Sal","Sev","Maj2","Maj1","Mod","Min2","Min1","Typ")

ames$Functional <- factor(ames$Functional, levels = lvl_functional, ordered = TRUE)
ames$Functional <- as.integer(ames$Functional)


# Land_Slope, Bsmt_Exposure, BsmtFin_Type_1, BsmtFin_Type_2, Garage_Finish, Paved_Drive

# Land_Slope
lvl_land_slope <- c("Gtl","Mod","Sev")

ames$Land_Slope <- factor(ames$Land_Slope, levels = lvl_land_slope, ordered = TRUE)
ames$Land_Slope <- as.integer(ames$Land_Slope)

# Bsmt_Exposure
lvl_bsmt_exposure <- c("No_Basement","No","Mn","Av","Gd")

ames$Bsmt_Exposure <- factor(ames$Bsmt_Exposure, levels = lvl_bsmt_exposure, ordered = TRUE)
ames$Bsmt_Exposure <- as.integer(ames$Bsmt_Exposure)

# BsmtFin_Type_1
lvl_bsmtfin1 <- c("No_Basement","Unf","LwQ","Rec","BLQ","ALQ","GLQ")

ames$BsmtFin_Type_1 <- factor(ames$BsmtFin_Type_1, levels = lvl_bsmtfin1, ordered = TRUE)
ames$BsmtFin_Type_1 <- as.integer(ames$BsmtFin_Type_1)

# BsmtFin_Type_2
lvl_bsmtfin2 <- c("No_Basement","Unf","LwQ","Rec","BLQ","ALQ","GLQ")

ames$BsmtFin_Type_2 <- factor(ames$BsmtFin_Type_2, levels = lvl_bsmtfin2, ordered = TRUE)
ames$BsmtFin_Type_2 <- as.integer(ames$BsmtFin_Type_2)

# Garage_Finish
lvl_garage_finish <- c("No_Garage","Unf","RFn","Fin")

ames$Garage_Finish <- factor(ames$Garage_Finish, levels = lvl_garage_finish, ordered = TRUE)
ames$Garage_Finish <- as.integer(ames$Garage_Finish)

# Paved_Drive
lvl_paved <- c("Dirt_Gravel","Partial_Pavement","Paved")

ames$Paved_Drive <- factor(ames$Paved_Drive, levels = lvl_paved, ordered = TRUE)
ames$Paved_Drive <- as.integer(ames$Paved_Drive)


# Utilities, Condition_2 y Paved se eliminan por la bajisima variabilidad que tienen
prop.table(table(ames$Utilities))
prop.table(table(ames$Condition_2))
prop.table(table(ames$Street))

ames <- ames %>% select(-c(Utilities, Street, Condition_2))



# Veamos las variables categoricas nominales
fac_cols <- names(ames)[sapply(ames, is.factor)]
fac_cols

# Para las categoricas nominales se crean dummys
#install.packages("fastDummies")
library(fastDummies)

fac_cols <- c("MS_SubClass","MS_Zoning","Lot_Shape","Land_Contour","Lot_Config",
              "Neighborhood","Condition_1","Bldg_Type","House_Style","Roof_Style",
              "Roof_Matl","Exterior_1st","Exterior_2nd","Mas_Vnr_Type","Foundation",
              "Heating","Central_Air","Electrical","Garage_Type","Sale_Type","Sale_Condition")

ames_dum <- fastDummies::dummy_cols(
  ames,
  select_columns = fac_cols,
  remove_selected_columns = TRUE,   # quita las originales
  remove_first_dummy = TRUE,       # TRUE evita colinealidad (dummy trap)
  ignore_na = FALSE                 # crea dummy para NA si existieran
)

# veamos la suma de variables factor que quedan
fac_cols <- names(ames_dum)[sapply(ames_dum, is.factor)]
length(fac_cols)  


# La base de datos ya lista con las dummies y todo es ames_dum

# Tipos de cada columna
var_types <- sapply(ames_dum, class)

# Conteo por tipo
table(var_types)



# Var types antes de hacer dummies
var_types_orig <- sapply(ames, class)
table(var_types_orig)


# 21 Variables se convierten en 161 al codificar, hay que ver pesos del modelo
# Lo ideal es poder eliminar las que sean necesarias y si computancionalmente 
# es muy caro tal vez hacer reducción de dimensionalidad con PCA
# Veamos si hay NA en la base de datos
colSums(is.na(ames_dum))
View(ames)



library(moments)
# Calcular skewness
skew_value <- skewness(ames$Sale_Price, na.rm = TRUE)

# Gráfico estilo seaborn
ggplot(ames, aes(x = Sale_Price)) +
  geom_histogram(aes(y = ..density..), 
                 bins = 40, 
                 fill = "steelblue", 
                 color = "white", 
                 alpha = 0.7) +
  geom_density(color = "navy", size = 1.2) +
  labs(title = "SalePrice Distribution",
       subtitle = paste("Skewness =", round(skew_value, 3)),
       x = "SalePrice", 
       y = "Density") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    panel.grid.minor = element_blank()
  )

# Calcular skewness
skew_value <- skewness(ames$Sale_Price_Log, na.rm = TRUE)

# Gráfico estilo seaborn
ggplot(ames, aes(x = Sale_Price_Log)) +
  geom_histogram(aes(y = ..density..), 
                 bins = 40, 
                 fill = "steelblue", 
                 color = "white", 
                 alpha = 0.7) +
  geom_density(color = "navy", size = 1.2) +
  labs(title = "SalePriceLog Distribution",
       subtitle = paste("Skewness =", round(skew_value, 3)),
       x = "SalePrice", 
       y = "Density") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    panel.grid.minor = element_blank()
  )



# FINALMENTE, EXPORTANDO LA BASE DE DATOS ames_dum SE TIENE LISTA PARA TRABAJAR


