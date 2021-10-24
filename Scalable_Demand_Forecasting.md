Scalable Demand Forecasting and Safety Stock Modelling using Forecast
Error
================
Maciej Lecicki
24 10 2021

<br/> <br/> <br/>

### Purpose and objectives

<br/>

The purpose of this project is to illustrate how to develop robust
demand forecasting model based on time series analysis and machine
learning in R environment. <br/>

Important goal of this project is to show: <br/> - potential of open
source software like R for demand forecasting,<br/> - scalability of
this approach, namely ease of forecasting automation for multiple
products/regions,<br/> - ability to shape model selection criteria using
common and customized metrics,<br/> - focus on forecast error and
confidence intervals as input to scenario planning for supply planning
or S&OP process,<br/> - use of forecast error as ‘demand uncertainty’
part of Safety Stock calculation.

##### Libraries and data

<br/>

``` r
library('tidyverse')
library('tidymodels')
library('modeltime')
library('timetk')
library('lubridate')
library('workflowsets')
library('tune')
library('patchwork')
```

Key libraries used in this project are modeltime and tidymodels, both
based on ‘tidyverse’ principles.

``` r
data <- read_csv('data/elecsupply.csv')
glimpse(data)
```

    ## Rows: 173
    ## Columns: 13
    ## $ date        <chr> "01.01.1990", "01.02.1990", "01.03.1990", "01.04.1990~
    ## $ Belgium     <dbl> 6287, 5546, 5975, 5461, 5272, 5171, 4735, 5132, 5293,~
    ## $ Denmark     <dbl> 3166, 2737, 2906, 2550, 2532, 2398, 2116, 2635, 2587,~
    ## $ Greece      <dbl> 3313, 2837, 3017, 2714, 2909, 2940, 3215, 3040, 2577,~
    ## $ Spain       <dbl> 13986, 11785, 12767, 11805, 11693, 12254, 12900, 1150~
    ## $ Ireland     <dbl> 1326, 1225, 1292, 1189, 1162, 1077, 1084, 1098, 1150,~
    ## $ Italy       <dbl> 22833, 20379, 22035, 20113, 20820, 20504, 21495, 1726~
    ## $ Netherlands <dbl> 7182, 6372, 6886, 6319, 6521, 6372, 6322, 6586, 6589,~
    ## $ Austria     <dbl> 4860, 4157, 4130, 4044, 3823, 3766, 3818, 3696, 3917,~
    ## $ Portugal    <dbl> 2867, 2214, 2338, 2157, 2273, 2208, 2387, 2090, 2235,~
    ## $ Finland     <dbl> 6688, 5536, 5927, 5176, 4969, 4262, 4355, 4820, 5064,~
    ## $ Sweden      <dbl> 15112, 12774, 13682, 11767, 10687, 9246, 8309, 9779, ~
    ## $ UK          <dbl> 29036, 29184, 34522, 24910, 22648, 27986, 21600, 2136~

``` r
range(data$date)
```

    ## [1] "01.01.1990" "01.12.2003"

Dataset consist of monthly demand for electrical supply for 12 European
countries collected between 1990 and 2003.

Let’s change table format from wide to long and visualize demand.

``` r
(elecsupply <- data %>%
  gather(country, elec_demand, Belgium:UK) %>%
  mutate(date = as_date(date, format = '%d.%m.%Y')) %>%
  relocate(country)
)
```

    ## # A tibble: 2,076 x 3
    ##    country date       elec_demand
    ##    <chr>   <date>           <dbl>
    ##  1 Belgium 1990-01-01        6287
    ##  2 Belgium 1990-02-01        5546
    ##  3 Belgium 1990-03-01        5975
    ##  4 Belgium 1990-04-01        5461
    ##  5 Belgium 1990-05-01        5272
    ##  6 Belgium 1990-06-01        5171
    ##  7 Belgium 1990-07-01        4735
    ##  8 Belgium 1990-08-01        5132
    ##  9 Belgium 1990-09-01        5293
    ## 10 Belgium 1990-10-01        5869
    ## # ... with 2,066 more rows

For demand visalization we could use plot_time_series() function (code
is included as comments) from timetk package. It provides interactive
element and has other advantages too, however I prefer ‘good old’ gglot2
which I find more elegant alternative if interaction isn’t necessary.

``` r
# elecsupply %>%
#   group_by(country) %>%
#   plot_time_series(date, elec_demand, .facet_ncol = 4, .smooth = FALSE)

elecsupply %>%
  ggplot(aes(x = date, y = elec_demand, group = country, color = country)) +
      geom_line(show.legend = FALSE) +
      theme_minimal() +
  facet_wrap(. ~ country, ncol = 3, scales = 'free_y') +
  geom_smooth(method = 'lm', se = FALSE, show.legend = FALSE)
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

With quick glance at all plots we can tell that demand is seasonal and
there’s upward trend in demand for electrical supply.

<br/> ##### Time series modelling <br/>

The goal we’d like to achieve is development of best fit demand forecast
using available resources in R. The challenge is that we’d like to
automate this process and do it at scale, meaning that we’d like to
develop best models for all countries. <br/>

To do that, we’ll use modeltime package supported by tidymodels and
tidyverse. Let’s start by describing high level the process and list key
steps.<br/>

1.  Dataset will be split into train and test:<br/>

-   train will be used to build the model,<br/>
-   testset of 24 months will be used to validate acccuracy and select
    best model,<br/>
-   in addition, test will be used to evaluate expected error that can
    be cascaded to Safety Stock calculation and scenario planning,<br/>

2.  We’ll also expand dataset by 12 months to forecast demand for
    electrical supply using best model.
3.  We’ll explore ‘classic’ time series approach and machine learning to
    develop demand forecasting model,<br/>
4.  We’ll look into available error metrics and enrich them by custom
    metrics.<br/>

Findings at each step of the process will be visualized using ggplot2.
This can also be used as input to Shiny web app (if required to be
shared with user that doesn’t have access to R environment).

##### Train, test and future datasets

``` r
nested_data_tbl <- elecsupply %>%
  group_by(country) %>%
  extend_timeseries(
    .id_var = country,
    .date_var = date,
    .length_future = 12 # this will be our forecasting horizon (in months)
  ) %>%
  nest_timeseries(
    .id_var = country,
    .length_future = 12
  ) %>%
  split_nested_timeseries(
    .length_test = 24 # test dataset (periods are months)
  )

nested_data_tbl
```

    ## # A tibble: 12 x 4
    ##    country     .actual_data       .future_data      .splits         
    ##    <chr>       <list>             <list>            <list>          
    ##  1 Belgium     <tibble [173 x 2]> <tibble [12 x 2]> <split [149|24]>
    ##  2 Denmark     <tibble [173 x 2]> <tibble [12 x 2]> <split [149|24]>
    ##  3 Greece      <tibble [173 x 2]> <tibble [12 x 2]> <split [149|24]>
    ##  4 Spain       <tibble [173 x 2]> <tibble [12 x 2]> <split [149|24]>
    ##  5 Ireland     <tibble [173 x 2]> <tibble [12 x 2]> <split [149|24]>
    ##  6 Italy       <tibble [173 x 2]> <tibble [12 x 2]> <split [149|24]>
    ##  7 Netherlands <tibble [173 x 2]> <tibble [12 x 2]> <split [149|24]>
    ##  8 Austria     <tibble [173 x 2]> <tibble [12 x 2]> <split [149|24]>
    ##  9 Portugal    <tibble [173 x 2]> <tibble [12 x 2]> <split [149|24]>
    ## 10 Finland     <tibble [173 x 2]> <tibble [12 x 2]> <split [149|24]>
    ## 11 Sweden      <tibble [173 x 2]> <tibble [12 x 2]> <split [149|24]>
    ## 12 UK          <tibble [173 x 2]> <tibble [12 x 2]> <split [149|24]>

Actual and future data is nested inside newly created object and grouped
by country. In addition we have information about split for train and
test sets.<br/>

To get inside nested data we can use unnest() function from tidyr
library. Example is shown below. I encourage to also run commented code
to see the difference in output.

``` r
nested_data_tbl %>%
  filter(country == 'Denmark') %>%
  unnest(.actual_data) %>%
  select(country:elec_demand)
```

    ## # A tibble: 173 x 3
    ##    country date       elec_demand
    ##    <chr>   <date>           <dbl>
    ##  1 Denmark 1990-01-01        3166
    ##  2 Denmark 1990-02-01        2737
    ##  3 Denmark 1990-03-01        2906
    ##  4 Denmark 1990-04-01        2550
    ##  5 Denmark 1990-05-01        2532
    ##  6 Denmark 1990-06-01        2398
    ##  7 Denmark 1990-07-01        2116
    ##  8 Denmark 1990-08-01        2635
    ##  9 Denmark 1990-09-01        2587
    ## 10 Denmark 1990-10-01        2850
    ## # ... with 163 more rows

``` r
# nested_data_tbl %>%
#   filter(country == 'Denmark') %>%
#   unnest(.actual_data)
```

Nesting data in a form of tibble is very convenient and helpful from the
point of view of model scalability. This concept will be used heavily in
this project.

##### Time series modelling

<br/>

First, we’ll build demand forecasting model using Machine Learning.
Functions from modeltime library will be used in conjunction with
tidymodels package.<br/>

Machine Learning based models don’t recognize date based information,
hence we need to recode date into ‘ML friendly’ calendar based
qualitative features. Before that, to follow tidymodels standard we’ll
prepare recipe used in the model.<br/> There’s also a few other
important steps taken in below code related to data pre-processing. All
steps have relevant comments.

``` r
rec_xgb <- recipe(elec_demand ~ ., extract_nested_train_split(nested_data_tbl)) %>%
  step_timeseries_signature(date) %>% # create dummy calendar features based on date
  step_rm(date) %>% # remove date column - not require anymore
  step_zv(all_predictors()) %>% # remove zero value predictors (if there are any)
  step_dummy(all_nominal(), one_hot = TRUE) # dummy predictors that are character data
```

We can check our recipe using bake() formula.

``` r
bake(prep(rec_xgb), extract_nested_train_split(nested_data_tbl))
```

    ## # A tibble: 149 x 37
    ##    elec_demand date_index.num date_year date_year.iso date_half
    ##          <dbl>          <dbl>     <int>         <int>     <int>
    ##  1        6287      631152000      1990          1990         1
    ##  2        5546      633830400      1990          1990         1
    ##  3        5975      636249600      1990          1990         1
    ##  4        5461      638928000      1990          1990         1
    ##  5        5272      641520000      1990          1990         1
    ##  6        5171      644198400      1990          1990         1
    ##  7        4735      646790400      1990          1990         2
    ##  8        5132      649468800      1990          1990         2
    ##  9        5293      652147200      1990          1990         2
    ## 10        5869      654739200      1990          1990         2
    ## # ... with 139 more rows, and 32 more variables: date_quarter <int>,
    ## #   date_month <int>, date_month.xts <int>, date_wday <int>,
    ## #   date_wday.xts <int>, date_qday <int>, date_yday <int>,
    ## #   date_mweek <int>, date_week <int>, date_week.iso <int>,
    ## #   date_week2 <int>, date_week3 <int>, date_week4 <int>,
    ## #   date_month.lbl_01 <dbl>, date_month.lbl_02 <dbl>,
    ## #   date_month.lbl_03 <dbl>, date_month.lbl_04 <dbl>, ...

We can see that step_series_signature() formula created many calendar
features and ALL of them have been passed to recipe.<br/> At this point
we could at a question if we need all of them and if all of them make
sense taking into account granularity (frequency) of original date and
its format (1st day of the month, month and year) or would they rather
confuse our machine learning model?<br/> I think it’s the latter and
therefore let’s narrow features to correct ones taking into above.<br/>

TO BE CONTINUED.
