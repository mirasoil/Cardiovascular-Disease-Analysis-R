library(rpart)                                                                  #construirea arborilor de clasificare ??i regresie
library(rpart.plot)                                                             #plot an rpart model
library(tidyverse)                                                              #for ggplot, tibble, tidyr
library(rsample)                                                                #metoda de esantionare
library(caret)                                                                  #confusionMatrix
library(partykit)                                                               #reprezentarea modelelor de regresie si arbori de clasificare
library(ggplot2)                                                                #creare de grafice (tidyverse)
library(dplyr)                                                                  #functii precum mutate, filter, select
library(corrplot)                                                               #vizualizarea matricilor de corelatie
library(modelr)                                                                 #pipeline
library(pROC)                                                                   #curba ROC

#incarcam setul de date
cardio_train <- read_delim("cardio_train.csv", ";", escape_double = FALSE, trim_ws = TRUE)
View(cardio_train)

#eliminam coloana id
cardio_train = cardio_train %>% select(-id) %>% select(age, everything())  

#convertirea coloanei age din zile in ani
cardio_train$age <- as.numeric(cardio_train$age) %/% 365.25

#tensiunea cu valori de 1000 resp 1100 se considera 100 resp 110
cardio_train$ap_lo[cardio_train$ap_lo == 1000] <- 100
cardio_train$ap_lo[cardio_train$ap_lo == 1100] <- 110

#valori normale ale tensiunii, inaltimii, greutatii
cardio_train <-  subset(cardio_train, ap_hi %in% (90:240) 
                        & ap_lo %in% (50:190) 
                        & height %in% (120:200) 
                        & weight %in% (40:150))

#stergerea inregistrarilor unde ap_lo > ap_hi
cardio_train<-cardio_train[!(cardio_train$ap_lo>=cardio_train$ap_hi),]
view(cardio_train)

#corelatia atributelor
M <- cor(select_if(cardio_train, is.numeric)) #calculeaza coeficientul de corelatie
df <- data.frame(cardio_train)
corrplot(M,method = "number") 

library(ggcorrplot)   #vizualizarea matricei de corelatie
df <- data.frame(cardio_train)
model.matrix(~0+., data=df) %>% 
  cor(use="complete.obs") %>% 
  ggcorrplot(show.diag = F, type="lower", lab=TRUE, lab_size=5)


#M/F
cardio_train <- cardio_train %>% 
  mutate(gender = ifelse(gender == 1, "Woman", "Man")) 
#cholesterol + glu
cardio_train <- cardio_train %>% 
  mutate(cholesterol = ifelse(cholesterol >= 2, "Above normal", "Normal")) %>%
  mutate(gluc = ifelse(gluc >= 2, "Above normal", "Normal"))
#smoke, alco, active, cardio "Yes"/"No"
cardio_train <- cardio_train %>% 
  mutate(smoke = ifelse(smoke == 0, "No", "Yes")) %>%
  mutate(active = ifelse(active == 0, "No", "Yes")) %>% 
  mutate(alco = ifelse(alco == 0, "No", "Yes")) %>% 
  mutate(cardio = ifelse(cardio == 0, "No", "Yes")) 

#bmi
cardio_train$bmi <- with(cardio_train, weight/((height/100)^2))
#pulse
cardio_train$pulse <- with(cardio_train,  ap_hi-ap_lo)

#factors
cardio_train = cardio_train %>%
  mutate(
    gender = factor(gender),
    smoke = factor(smoke),
    active = factor (active),
    alco = factor(alco),
    cholesterol = factor ( cholesterol),
    gluc = factor(gluc)
  )

cardio_train <- cardio_train %>% mutate(cardio = factor(cardio))
table(cardio_train$cardio)   #34571 - 34487 


#distributia valorilor atributelor persoanelor cardiace
cardio_train %>%
  filter(cardio == "Yes") %>%
  select_if(is.numeric) %>%          
  gather(metric, value) %>%              
  ggplot(aes(value, fill=metric)) +      
  geom_density(show.legend = FALSE) +    
  facet_wrap(~metric, scales = "free")

#distributia valorilor atributelor persoanelor sanatoase
cardio_train %>%
  filter(cardio == "No") %>%
  select_if(is.numeric) %>%          
  gather(metric, value) %>%              
  ggplot(aes(value, fill=metric)) +      
  geom_density(show.legend = FALSE) +    
  facet_wrap(~metric, scales = "free")

hist(cardio_train$height)
hist(cardio_train$weight)

#femeile care fumeaza vs barbatii care fumeaza
library(janitor)                                                                #curatarea/explorarea datelor
tabyl(cardio_train, gender, smoke)                                              #tabelarea variabilelor

#color palette
library(wesanderson)

#Distributia fumatorilor pe sex
ggplot(cardio_train, aes(x = gender, fill = smoke)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..), vjust=-1)  +                    #stat-height of the bar proportional to the number of cases
  labs(title= "Distributia fumatorilor pe sex", y="Numarul pacientilor", x = "Sex", fill="Fumator") +
  scale_fill_manual(values = c("#e9d4ff", "#b491d9", "#56B4E9"), 
                    labels = c("Nefumator", "Fumator")) + 
  scale_x_discrete(labels = c('Barbat','Femeie'))


#Distributia colesterolului pe sex
ggplot(cardio_train, aes(x = cholesterol, fill = gender)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..), vjust=-1)  + 
  labs(title= "Distributia colesterolului pe sex", y="Numarul pacientilor", x = "Nivelul colesterolului", fill="Sex") +
  scale_fill_manual(values = c("#e9d4ff", "#b491d9", "#56B4E9"), 
                    labels = c("Feminin", "Masculin"))                          #exista un numar mai mare de femei cu nivelul colesterolului peste limita
#reprezentare tabelara
tabyl(cardio_train, gender, cholesterol)





########################################
           #Naive Bayes
########################################

#impartire: 70% pentru antrenament si 30% pentru test
set.seed(123)                                                                   #setam un seed pentru a avea aceleasi rezultate si daca rulam script-ul la o data ulterioara (rezultate reproductibile)

#train - 4 parametrii: setul de date, metoda de invatare (naive bayes), trcontrol (ce procedura alegem pentru validare), tunegrid
split <- initial_split(cardio_train, prop = 0.7, strata="cardio")               #impartim setul cu stratificare dupa cardio, in 70%-30%
train <- training(split)                                                        #genereaza 48341 de instante in setul de training (70%)
test <- testing(split)

table(train$cardio)                                                             #24200/48341=50.06% - pastreaza proportia de No si pe setul de antrenament
table(test$cardio) 


#cardio este clasa tinta Y iar restul de 13 sunt X(independente)
features <- setdiff(names(train), "cardio")                                     #setul de features=toate atributele din setul antrenament care nu sunt cardio - 13 atribute
x <- train[,features]                                                           #48341 de instante doar cu atr features
y <- train$cardio                                                               #variabila dependenta

#metoda de validare (Cross-Validation cu 10-folds)
fitControl <- trainControl(             
  method = "cv",
  number = 10
)

modNbSimpleCV <- train(                                                         #invatarea modelului
  x = x,                                                                        #set de date cu var independente
  y = y,                                                                        #set de date cu var dependenta - var de iesire, prezisa
  method = "nb",                        
  trControl = fitControl                                                        #va face 10 modele - media lor 
)
modNbSimpleCV
confusionMatrix(modNbSimpleCV)

#cautam printr-o cautare extensiva care combinatie dintre parametrii e cea mai buna
searchGrid <- expand.grid(
  usekernel = c(TRUE, FALSE),                                                   #folosim sau nu kernel - poate lua 2 valori
  fL = 0.5,                                                                     #la variabilele nominale nu pornim de la 0 ci de la 0.5 - poate lua o valoare
  adjust = seq(0, 5, by = 1)                                                    #vom ajusta kernel cu cate un pas -poate lua 6 valori
)

#mai adaugam un parametru - tuneGrid
modNbCVSearch <- train(                  
  x = x,
  y = y,
  method = "nb",
  trControl = fitControl,
  tuneGrid = searchGrid                                                         #construieste un model pentru fiecare iteratie posibila              
)
modNbCVSearch
confusionMatrix(modNbCVSearch)                                                  #Accuracy (average) : 0.7207

#topul celor mai bune modele
modNbCVSearch$results %>%
  top_n(5, wt = Accuracy) %>%
  arrange(desc(Accuracy))

#vom face predictii pe cel mai bun model
pred <- predict(modNbCVSearch, test)                                            #il vom testa pe date noi - datele de test (ia modelul 1 si aplica acest model pe datele de test)
predProb <- predict(modNbCVSearch, test, type = "prob")                         #dorim sa vedem si probabilitatea pentru fiecare valoare- predictie cu probabilitati
confusionMatrix(pred, test$cardio)                                              #Accuracy : 0.7182   Specificity : 0.6248


#curba ROC
dataset <- data.frame(                             
  actual.class <- test$cardio,                                                  #furnizam prima coloana cu valorile reale
  probability <- predProb[,1]                                                   #iar a doua coloana cu probabilitatea pentru clasa pozitiva
)
roc.val <- roc(actual.class ~ probability, dataset)
roc.val       #AUC = 0.7836

adf <- data.frame(
  specificity <- 1-roc.val$specificities,                                       #false positives pe axa X
  sensitivity <- roc.val$sensitivities)
ggplot(adf, aes(specificity, sensitivity)) +
  geom_line(color = 'blue') 

#train one model on the training set and apply it on the test set
searchOne <- expand.grid(                                                       #modelul cel mai bun (cu kernel true, fL 0.5 si adjust 4)
  usekernel = TRUE,
  fL = 0.5,
  adjust = 4
)
fitControlNone <- trainControl(                                                 #antreneaza un model pe tot setul de antrenament fara cross-validation
  method = "none"
)


modNbNone <- train(
  x = x,
  y = y,
  method = "nb",
  trControl = fitControlNone,
  tuneGrid = searchOne
)
modNbNone

predNone <- predict(modNbNone, test)                                            #predictii pe setul de test
confusionMatrix(predNone, test$cardio)                                          #Accuracy : 0.7182 Specificity : 0.6248


#curba ROC
pred_mod_none <- predict(modNbNone, test, type="prob")
dataset_none <- data.frame(                             
  actual.class <- test$cardio,                                                  #furnizez prima coloana cu valorile reale
  probability <- pred_mod_none[,1]                                              #iar a doua coloana cu probabilitatea pentru clasa pozitiva
)
roc.val <- roc(actual.class ~ probability, dataset_none)
roc.val       #AUC = 0.7836

adf <- data.frame(
  specificity <- 1-roc.val$specificities,                                       #false positives pe axa X
  sensitivity <- roc.val$sensitivities)
ggplot(adf, aes(specificity, sensitivity)) +
  geom_line(color = 'blue')




#optimizare dupa curba ROC
fitControlROC <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,                                                            #pastreaza probabilitatile claselor
  summaryFunction = twoClassSummary,
  savePredictions = TRUE                                                        #pastreaza predictiile
)

#metric = ROC specifica dupa ce dorim optimizarea modelului
modNbSearchROC = train(
  x = x, 
  y = y,
  method = "nb",
  trControl = fitControlROC,
  tuneGrid = searchGrid,
  metric = "ROC"
)
modNbSearchROC
confusionMatrix(modNbSearchROC)                                                 #Accuracy (average) : 0.7181
modNbSearchROC$results %>%
  top_n(5, wt = ROC) %>%
  arrange(desc(ROC))


predROC <- predict(modNbSearchROC, test)
predProbROC <- predict(modNbSearchROC, test, type = "prob")
confusionMatrix(predROC, test$cardio)                                           #Accuracy : 0.7149  Specificity : 0.6056

#curba ROC
dataset1 <- data.frame(                             
  actual.class <- test$cardio,                                                  #furnizam prima coloana cu valorile reale
  probability <- predProbROC[,1]                                                #iar a doua coloana cu probabilitatea pentru clasa yes
)
roc.val1 <- roc(actual.class ~ probability, dataset1)
roc.val1                                                                        #AUC = 78.5%  vs 78.63%

adf1 <- data.frame(
  specificity <- 1-roc.val1$specificities,                                      #false positives pe axa X
  sensitivity <- roc.val1$sensitivities)
ggplot(adf1, aes(specificity, sensitivity)) +
  geom_line(color = 'blue') 




########################################
        #Regresie logistica
########################################

#Distributia tensiunii pentru cardio
ggplot(cardio_train) +
  geom_point(aes(x = ap_lo, y = ap_hi, color = cardio, shape = cardio)) +
  scale_shape_manual(values = c(1,4))

#Boxplot pentru cardio
ggplot(cardio_train) + 
  geom_boxplot(aes(x = cardio, y = ap_hi, fill = cardio))


by_cardio <-group_by(cardio_train, cardio)                                      #am grupat setul de date in functie de variabila cardio
summarize(by_cardio, count = n())                                               #numarul persoanelor sanatoase si numarul cardiacilor

#regresie logistica dupa tensiunea sistolica
mod <- glm(data = cardio_train, cardio ~ ap_hi, family = binomial)              
summary(mod)                                                                    #cresterea valorii tensiunii creste probabilitatea de a fi cardiac

grid <- cardio_train %>%
  data_grid(ap_hi = seq_range(ap_hi, 20)) %>%                                   #20 de valori din aria tensiunii              
  add_predictions(mod, "prob_default", type = "response")

ggplot() +
  geom_line(data = grid, aes(ap_hi, prob_default), color = "red", size = 1)     #curba S

nd <- tribble(~ap_hi, 120, 160)                                                 #pentru ap_hi cu nivel fixat - sansele de a fi cardiac
predicted <- predict(mod, newdata = nd, type = "response")
predicted                                                                       #pentru o valoare a tensiunii de 120 avem 39% sanse ca persoana sa sufere de CVD, iar pentru 160 - 91% sanse


#colesterol
mod_chol <- glm(data = cardio_train, cardio ~ cholesterol, family = binomial)        
summary(mod_chol)                                                               #o persoana cu nivelul colesterolului normal are sanse scazute de a fi cardiaca

nd_chol <- tribble(~cholesterol, "Above normal", "Normal")
predicted <- predict(mod_chol, newdata = nd_chol, type = "response")
predicted                                                                       #perosanele cu nivelul colesterolului peste limita au 67% sanse sa fe cardiace, in timp ce persoanele cu un nivel normal doar 43%


#Corelatia cardio-sex-varsta
mod_age_gender <- glm(data = cardio_train, cardio ~ age + gender, family = binomial)
summary(mod_age_gender)                                                         #pentru un nivel al varstei fixat, faptul ca un pacient este femeie poate scadea sansele de a avea boli cardiovasculare

#Sunt barbatii mai predispusi la CVD decat femeile ?
grid_woman <- cardio_train %>%                                                  #construiesc un set de date nou
  data_grid(age = seq_range(age, 5)) %>%                              
  mutate(gender = "Woman") %>%   
  add_predictions(mod_age_gender, "prob_default", type = "response")            #facem predictii

grid_man <- cardio_train %>%                                                
  data_grid(age = seq_range(age, 5)) %>%                              
  mutate(gender = "Man") %>%                                                   
  add_predictions(mod_age_gender, "prob_default", type = "response")

ggplot(grid_woman, aes(age, prob_default)) +
  geom_line(color = "blue", size = 2) +
  geom_line(data = grid_man, color = "orange", size = 2)                        #barbatii sunt mai predispusi la CVD, insa diferenta minuscula datorita setului de date echilibrat

#Distributia cardio dupa sex si varsta
ggplot(cardio_train, aes(x=cardio, y=age, fill=gender)) + 
  geom_boxplot()                                                                #femeile se imbolnavesc putin mai tarziu decat barbatii, insa media varstei acestora e la fel


#antrenarea modelului
set.seed(123)                            

#antrenare model reg log cu cv = 10 folds
trCntl <- trainControl(method = "CV",number = 10)
glmModel <- train(cardio ~ ., data = train, trControl = trCntl, method="glm", family = "binomial")
confusionMatrix(glmModel)

#matricea de confuzie
predictie <- predict(glmModel, test)
confusionMatrix(predictie,test$cardio)                                          #Accuracy = 0.7253  Specificity = 0.6732

#fara cv 10 folds
mod_all_train <- glm(data = train, cardio ~ ., family = binomial)
summary(mod_all_train)

#treshold 0.5
pred_test <- predict(mod_all_train, newdata = test, type = "response")
t = table(pred_test > 0.5, test$cardio)                                         #pun treshold-ul, cei peste 0.5 sa-i prezica potential de dat faliment
t

spec = t[2,2]/(t[2,2]+t[1,2])
spec  #67.32%
sens = t[1,1]/(t[1,1]+t[2,1])
sens  #77.72%

#treshold 0.3 - gaseseste 91% din cei bolnavi
pred_test_mod <- predict(mod_all_train, newdata = test, type = "response")
t=table(pred_test_mod > 0.3, test$cardio)                                       #in functie de cum setam treshold-ul se imbunatateste modelul
t

spec = t[2,2]/(t[2,2]+t[1,2])
spec  #91.44%
sens = t[1,1]/(t[1,1]+t[2,1])
sens  #39.69%

sp = c()
sn = c()

#calculam toate valorile pentru senzitivitate si specificitate
for(prob in seq(0,1,by=0.0001)){
  t=table(pred_test >= prob, test$cardio)
  if(dim(t)[1]==2){
    spec = t[2,2]/(t[2,2]+t[1,2])
    sens = t[1,1]/(t[1,1]+t[2,1])
    
    sp = c(sp, spec)
    sn = c(sn, sens)
  }
}
ROC  = data.frame(sp, sn)
names(ROC) = c("specificity","sensitivity")


#ROC+AUC - ca si la NB
dataset2 <- data.frame(                             
  actual.class <- test$cardio,                                                  #furnizam prima coloana cu valorile reale
  probability <- pred_test                                                      #iar a doua coloana cu probabilitatea prezise pentru clasa pozitiva 
)
roc.val2 <- roc(actual.class ~ probability, dataset2)
roc.val2     #AUC = 0.7935

adf2 <- data.frame(
  specificity <- 1-roc.val2$specificities,                                      #false positives pe axa X
  sensitivity <- roc.val2$sensitivities)
ggplot(adf2, aes(specificity, sensitivity)) +
  geom_line(color = 'blue') 




########################################
             #Arbori
########################################
#cu libraria rpart care foloseste eroarea ca si metrica de optimizare - cost-complexity optimization
set.seed(123)

table(c_train$cardio)
table(c_test$cardio)


#arbore de decizie
set.seed(123)

m1 = rpart(                                                                     #arborele de decizie
  formula = cardio ~ . ,                                                        #cardio in functie de toate celelalte
  data = train,                                                                 #pe setul de train - folosim doar 80% din ele (5 grupuri de date, 4 pentru antrenament si 1 pentru test dupa care face media)
  method = "class"                                                              #clasificare
)
m1
summary(m1)
plotcp(m1)
printcp(m1)

rpart.plot(m1)  #afisare arbore

pred_m1 <- predict(m1, newdata = test, target ="class")                         #pentru fiecare clasa din setul de test ne da predictiile pentru cele 2 clase
pred_m1 <- as_tibble(pred_m1) %>%
  mutate(class = ifelse(No >= Yes, "No", "Yes"))                                #predictie pe probabilitati - predictie pe clase Yes/No
pred_m1
table(pred_m1$class, test$cardio)
confusionMatrix(factor(pred_m1$class), factor(test$cardio)) #Accuracy : 0.7135 Sensitivity : 0.8024  Specificity : 0.6244
Specificitate <- 6460/(6460+3886)


#Curba ROC
pred_test_dt <- predict(m1, newdata = test, type = "prob")

dataset3 <- data.frame(                             
  actual.class <- test$cardio,                                                  #furnizam prima coloana cu valorile reale
  probability <- pred_test_dt[,1]                                               #iar a doua coloana cu probabilitatea prezisa pentru clasa pozitiva 
)
roc.val3 <- roc(actual.class ~ probability, dataset3)
roc.val3     #AUC = 0.7134

adf3 <- data.frame(
  specificity <- 1-roc.val3$specificities,                                      #false positives pe axa X
  sensitivity <- roc.val3$sensitivities)
ggplot(adf3, aes(specificity, sensitivity)) +
  geom_line(color = 'blue') 



#vizualizare fara ap_hi
arbori_train = cardio_train %>% select(-ap_hi) %>% select(age, everything())

m1 = rpart(                                                                     #arborele de decizie
  formula = cardio ~. ,                                                         #Sales in functie de toate celelalte
  data = arbori_train,                                                          #pe setul de train - folosim doar 80% din ele (5 grupuri de date, 4 pentru antrenament si 1 pentru test dupa care face media)
  method = "class"                                                              #clasificare
)
m1
summary(m1)
rpart.plot(m1)




#entropia
library(tree)                                                                   #gini si entropy, optimizare metrici prin taiere - atributul age devine relevant   - se doreste a fi cat mai mare
set.seed(123)
m1_tree <- tree(cardio ~., data = train)                                        #works with deviance computed with entropy
m1_tree                                                                         #construim arborele folosind entropia => arborele e mult mai mare
summary(m1_tree)

#afisare arbore
library(maptree)
draw.tree(m1_tree)

pred_m1_tree <- predict(m1_tree, newdata = test, target = "class")              #predictii
pred_m1_tree <- as_tibble(pred_m1_tree) %>% 
  mutate(class = ifelse(No >= Yes, "No", "Yes"))                                #atribuim clase
confusionMatrix(factor(pred_m1_tree$class), factor(test$cardio))                #Accuracy : 0.7135 Sensitivity : 0.8024  Specificity : 0.6244


#curba ROC
pred_entropy <- predict(m1_tree, newdata = test, target = "class")

dataset4 <- data.frame(                             
  actual.class <- test$cardio,                                     
  probability <- pred_entropy[,1]                            
)
roc.val4 <- roc(actual.class ~ probability, dataset4)
roc.val4     #AUC = 0.767

adf4 <- data.frame(
  specificity <- 1-roc.val4$specificities,            
  sensitivity <- roc.val4$sensitivities)
ggplot(adf4, aes(specificity, sensitivity)) +
  geom_line(color = 'blue') 





#cu gini
set.seed(123)
m1_tree_gini <- tree(cardio ~., data = train, split=c("deviance","gini"))       #works with Gini index
m1_tree_gini
summary(m1_tree_gini)  

pred_m1_tree_gini <- predict(m1_tree_gini, newdata = test, target = "class")
pred_m1_tree_gini <- as_tibble(pred_m1_tree_gini) %>% 
  mutate(class = ifelse(No >= Yes, "No", "Yes"))
confusionMatrix(factor(pred_m1_tree_gini$class), factor(test$cardio))           #Accuracy : 0.7135 Sensitivity : 0.8024  Specificity : 0.6244


#CURBA ROC
pred_gini <- predict(m1_tree_gini, newdata = test, target = "class")

dataset5 <- data.frame(                             
  actual.class <- test$cardio,                                     
  probability <- pred_gini[,1]                                     
)
roc.val5 <- roc(actual.class ~ probability, dataset5)
roc.val5     #AUC = 0.767

adf5 <- data.frame(
  specificity <- 1-roc.val5$specificities,                                      #false positives pe axa X
  sensitivity <- roc.val5$sensitivities)
ggplot(adf5, aes(specificity, sensitivity)) +
  geom_line(color = 'blue') 






#BAGGING
library(ipred)
set.seed(123)
bagged_m1 <- bagging(cardio ~ .,
                     data = train, 
                     coob = TRUE)
bagged_m1  #OOB = 0.3176
summary(bagged_m1)    #err = 0.3175565
pred_bagged_m1 <- predict(bagged_m1, newdata = test, target = "class")
confusionMatrix(pred_bagged_m1, factor(test$cardio))  #Accuracy : 0.6965  Specificity : 0.6911  Sensitivity : 0.7020

#ROC + AUC
pred_bgd_m1 <- predict(bagged_m1, newdata = test, type = "prob")

dataset6 <- data.frame(                             
  actual.class <- test$cardio,                                                  #furnizam prima coloana cu valorile reale
  probability <- pred_bgd_m1[,2]                                                #iar a doua coloana cu probabilitatea prezisa pentru clasa pozitiva 
)
roc.val6 <- roc(actual.class ~ probability, dataset6)
roc.val6     #AUC = 0.753

adf6 <- data.frame(
  specificity <- 1-roc.val6$specificities,                                      #false positives pe axa X
  sensitivity <- roc.val6$sensitivities)
ggplot(adf6, aes(specificity, sensitivity)) +
  geom_line(color = 'blue') 



#antrenarea modelului
ntree <- seq(10, 50, by = 1)
misclassification <- vector(mode = "numeric", length = length(ntree))
for (i in seq_along(ntree)) {
  set.seed(123)
  model <- bagging( 
    cardio ~.,
    data = train,
    coob = TRUE,
    nbag = ntree[i])
  misclassification[i] = model$err
}
plot(ntree, misclassification, type="l", lwd="2")
#ceva mai mult de 48 bags sunt necesare pentru a stabiliza rata de eroare

bagged_m1_48 <- bagging(cardio ~ .,
                        data = train, 
                        coob = TRUE, 
                        nbag = 48)
bagged_m1_48
#summary(bagged_m1_48)
pred_bagged_m1_48 <- predict(bagged_m1_48, newdata = test, target = "class")
confusionMatrix(pred_bagged_m1_48, factor(test$cardio))                         #Accuracy : 0.7008 Sensitivity : 0.7060 Specificity : 0.6956 


#curba ROC pt bagging
pred_bagging_48 <- predict(bagged_m1_48, newdata = test, type = "prob")

dataset_bagging <- data.frame(                             
  actual.class <- test$cardio,                                                  #furnizam prima coloana cu valorile reale
  probability <- pred_bagging_48[,1]                                            #iar a doua coloana cu probabilitatea pentru clasa yes
)
roc.val_bagging <- roc(actual.class ~ probability, dataset_bagging)
roc.val_bagging                                                                 #AUC = 0.7576

adf_bagging <- data.frame(
  specificity <- 1-roc.val1$specificities,                                      #false positives pe axa X
  sensitivity <- roc.val1$sensitivities)
ggplot(adf_bagging, aes(specificity, sensitivity)) +
  geom_line(color = 'blue') 


