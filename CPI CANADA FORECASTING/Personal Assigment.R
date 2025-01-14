library(readxl)
library(tidyverse)
library(fpp3)
library(forecast)
library(lubridate)
CPI <- read_excel("C:/CONESTOGA/PREDICTIVE ANALYTICS 2/FORECASTING/PERSONAL ASSIGMENT/CPI.xlsx")
Price_index <- CPI %>%
  group_by(Year = year(REF_DATE), Month = month(REF_DATE, label = TRUE)) %>%
  summarise(TotalCPI = sum(VALUE), .groups = "drop") %>%
  ungroup()
# Create a time series object (assuming monthly data)
Price_ts <- ts(Price_index$TotalCPI, start = c(min(Price_index$Year), 1), frequency = 12)
Price_time_series <- Price_index %>%
  mutate(Date = yearmonth(paste(Year, Month))) %>%
  as_tsibble(index = Date)

#Plotting time-series
Price_time_series %>% autoplot(TotalCPI)

#Creating an ACF plot

Price_time_series %>% 
  ACF(TotalCPI, lag_max = 94) %>%
  autoplot() +
  labs(title = "Canadian CPI ACF") 

#Descomponing the time seires using three different models
dcmp <- Price_time_series %>%
  model(
    classical = classical_decomposition(TotalCPI, type = "additive"),
    x11 = X_13ARIMA_SEATS(TotalCPI ~ x11()),
    stl = STL(TotalCPI)
  )
x11comps <- dcmp %>% select(x11)%>%components(dcmp)
stlcomps <- dcmp %>% select(stl) %>% components(dcmp)
classcomps <- dcmp %>% select(classical)%>% components(dcmp)

stlcomps %>% tail()
x11comps%>% tail()
classcomps %>% tail()

x11comps%>%autoplot()
stlcomps %>% autoplot()
classcomps %>% autoplot()

#Predicting new TotalCPIs using NAVIE, MEAN, NAIVE DRIFT and naive seasonal adjust models

mean <- Price_time_series%>%
  model(MEAN(TotalCPI))

mean %>% forecast(h = 12) %>%autoplot(Price_time_series) +
  ggtitle('Mean Forecasting Model')

naive <- Price_time_series %>% model(NAIVE(TotalCPI ~ drift()))
naive %>% forecast(h =12) %>% autoplot(Price_time_series) +
  ggtitle("Naive Forecast Model")

seasonal <- Price_time_series %>% model(SNAIVE(TotalCPI))
seasonal %>% forecast(h = 12) %>% autoplot(Price_time_series) +
  ggtitle("Seasonal Naive Forescast Model")

seasonal %>% gg_tsresiduals()
seasonal%>% augment() %>%
  features(.innov, ljung_box, lag = 12, dof = 0)

fit_dcmp <- Price_time_series %>%
  model(stlf = decomposition_model(
    STL(TotalCPI),
    NAIVE(season_adjust)
  ))
fit_dcmp %>% forecast(h = 12) %>%
  autoplot(Price_time_series) +
  ggtitle("Fitted STL Forecasting")

#Creating training and test sets
Price_time_series_train <- Price_index %>%
  mutate(Date = yearmonth(paste(Year, Month))) %>%  
  filter(Year >= 2000 & Year <= 2018)

Price_time_series_test <- Price_index %>%
  mutate(Date = yearmonth(paste(Year, Month))) %>%
  filter(Year >= 2019)

Price_time_series_tsibble_train <- Price_time_series_train %>%
  as_tsibble(index = Date)
Price_time_series_tsibble_test<- Price_time_series_test %>%
  as_tsibble(index = Date)

#Creating new models using the train and test sets
CPI_fit <- Price_time_series_tsibble_train%>%
  model(
    MeanModel =MEAN(TotalCPI),
    NaiveModel = NAIVE(TotalCPI),
    DriftModel = NAIVE(TotalCPI ~ drift())
  )
#Forescasting the TotalCPIs
CPI_fc <- CPI_fit %>% forecast(h=12)
CPI_fc %>% autoplot(Price_time_series_tsibble_test, level = NULL) +
  ggtitle("Forescasting models with training and test sets")
#Measuring the accuracy of the model
accuracy(CPI_fc,Price_time_series_tsibble_test)%>%select(.model,RMSE,MAE)

#Using cross validation to tune the model
cross_validation_cpi <- Price_time_series%>%
  slice(-n())%>%
  stretch_tsibble(.init = 4)

cross_validation_cpi %>%
  model(
    M = MEAN(TotalCPI),
    D = NAIVE(TotalCPI ~ drift())
  ) %>%
  forecast (h = 1) %>%
  accuracy(Price_time_series)

#plotting the results
Results <- cross_validation_cpi %>%
  model(
    M = MEAN(TotalCPI),
    D = NAIVE(TotalCPI ~ drift())
  ) %>%
  forecast (h = 1)






