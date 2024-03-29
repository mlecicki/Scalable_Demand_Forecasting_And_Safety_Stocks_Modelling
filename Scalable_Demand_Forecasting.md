Scalable Demand Forecasting and Safety Stock Modelling using Forecast
Error
================
Maciej Lecicki
24.10.2021

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
library('wesanderson')
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

Dataset consists of monthly demand for electrical supply for 12 European
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

##### Time series modelling

<br/>

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
    electrical supply; it will be done by refitting best model on full
    dataset (train + test),<br/>
3.  We’ll explore ‘classic’ time series approach and machine learning
    based model to develop demand forecasting model which minimizes
    selected forecast error metric,<br/>
4.  We’ll look into standard error metrics available in R libraries and
    enrich metrics selection through development of own, custom
    metrics.<br/>

Findings at each step of the process will be visualized using ggplot2.
This can also be used as input to Shiny web app (if required to be
shared with user that doesn’t have access to R environment).

##### Train, test and future datasets

<br/>

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
library. Example is shown below. I encourage to also run commented out
code to see the difference in output.

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

Let’s re-write our recipe and remove unwanted columns in step_rm()
function, where we can list what needs to be de-selected, or by using !
and - operators, what needs to be kept. Please notice that since
character features are removed, one hot encoding is not needed anymore.

``` r
rec_xgb <- recipe(elec_demand ~ ., extract_nested_train_split(nested_data_tbl)) %>%
  step_timeseries_signature(date) %>%
  step_rm(!elec_demand, -c(date_year.iso:date_month)) %>%
  step_zv(all_predictors()) # remove zero value predictors (if there are any)
  # step_dummy(all_nominal(), one_hot = TRUE) # not required anymore
```

Let’s take a look corrected recipe.

``` r
bake(prep(rec_xgb), extract_nested_train_split(nested_data_tbl))
```

    ## # A tibble: 149 x 5
    ##    elec_demand date_year.iso date_half date_quarter date_month
    ##          <dbl>         <int>     <int>        <int>      <int>
    ##  1        6287          1990         1            1          1
    ##  2        5546          1990         1            1          2
    ##  3        5975          1990         1            1          3
    ##  4        5461          1990         1            2          4
    ##  5        5272          1990         1            2          5
    ##  6        5171          1990         1            2          6
    ##  7        4735          1990         2            3          7
    ##  8        5132          1990         2            3          8
    ##  9        5293          1990         2            3          9
    ## 10        5869          1990         2            4         10
    ## # ... with 139 more rows

By default, it returns train set for first aggregation field (country in
our case). It’s possible however to check recipe for any country as
shown below.

``` r
bake(prep(rec_xgb), extract_nested_train_split(nested_data_tbl %>%
                                                 filter(country == 'Denmark')))
```

    ## # A tibble: 149 x 5
    ##    elec_demand date_year.iso date_half date_quarter date_month
    ##          <dbl>         <int>     <int>        <int>      <int>
    ##  1        3166          1990         1            1          1
    ##  2        2737          1990         1            1          2
    ##  3        2906          1990         1            1          3
    ##  4        2550          1990         1            2          4
    ##  5        2532          1990         1            2          5
    ##  6        2398          1990         1            2          6
    ##  7        2116          1990         2            3          7
    ##  8        2635          1990         2            3          8
    ##  9        2587          1990         2            3          9
    ## 10        2850          1990         2            4         10
    ## # ... with 139 more rows

We now have recipe for Machine Learning. Classic time series algorithms
don’t require data preprocessing thus saving recipe to an object. <br/>
Next step is create workflows, which are containers that aggregate
information required to fit and predict from a model. We’ll build one
machine learning workflow and four others that cover more standard time
series based models.

1.  boosted_tree workflow.

Like any other ML algorithm, xgboost has hyperparameters.
Hyperparameters tuning is out of scope of this project, however we’ll
show how it’s done using tidymodels library. Detailed process can be
found under below link:<br/>
<https://towardsdatascience.com/dials-tune-and-parsnip-tidymodels-way-to-create-and-tune-model-parameters-c97ba31d6173>

Tuning is done using our recipe which is based on a subset of our data
(first country).

``` r
tune_spec <- boost_tree(
  learn_rate = tune(),
  trees = tune(),
  mtry = tune(),
  tree_depth = tune()
) %>%
  set_engine('xgboost') %>%
  set_mode('regression')
#
folds_3 <- vfold_cv(extract_nested_train_split(nested_data_tbl),
                    v = 3)

wflw_xgb_1 <- workflow() %>%
  add_model(tune_spec) %>%
  add_recipe(rec_xgb)

set.seed(300)
wflw_xgb_1_tune <- wflw_xgb_1 %>%
  tune_grid(
    resamples = folds_3,
    grid = 3
  )

wflw_xgb_1_tuned <- finalize_workflow(wflw_xgb_1,
                                      select_best(wflw_xgb_1_tune))

# optimal hyperparameters
wflw_xgb_1_tuned
```

    ## == Workflow ===============================================================
    ## Preprocessor: Recipe
    ## Model: boost_tree()
    ## 
    ## -- Preprocessor -----------------------------------------------------------
    ## 3 Recipe Steps
    ## 
    ## * step_timeseries_signature()
    ## * step_rm()
    ## * step_zv()
    ## 
    ## -- Model ------------------------------------------------------------------
    ## Boosted Tree Model Specification (regression)
    ## 
    ## Main Arguments:
    ##   mtry = 3
    ##   trees = 1085
    ##   tree_depth = 10
    ##   learn_rate = 0.00275270800743389
    ## 
    ## Computational engine: xgboost

Let’s use some of these values in the workflow.

``` r
wflw_xgb <- workflow() %>%
  add_model(boost_tree("regression",
                       mtry = 3,
                       trees = 1085,
                       tree_depth = 10,
                       learn_rate = 0.03) %>%
              set_engine("xgboost")) %>%
  add_recipe(rec_xgb)
```

2.  Temporal Hierarchical Forecasting (THIEF) workflow<br/>

All key information about it can be found under below link:<br/>
<https://robjhyndman.com/hyndsight/thief/>

As for the worklfow itself, notice difference in recipe between xgboost
and non-ML workflows.

``` r
wflw_thief <- workflow() %>%
  add_model(temporal_hierarchy() %>%
              set_engine("thief")) %>%
  add_recipe(recipe(elec_demand ~ ., extract_nested_train_split(nested_data_tbl)))
```

In above code, ‘elec_demand \~ .’ formula could be replaced by
‘elec_demand \~ date’ as date is the only predictor (historic demand).
The same applies to all remaining recipies in below workflows.

3.  Exponential smoothing workflow

``` r
wfl_exp_s <- workflow() %>%
  add_model(exp_smoothing() %>%
              set_engine("ets")) %>%
  add_recipe(recipe(elec_demand ~ date, extract_nested_train_split(nested_data_tbl)))
```

4.  (S)ARIMA workflow

``` r
wfl_arima <- workflow() %>%
  add_model(arima_reg() %>%
              set_engine("auto_arima")) %>%
  add_recipe(recipe(elec_demand ~ ., extract_nested_train_split(nested_data_tbl)))
```

auto_arima engine is used which means that autoregression, moving
average and seasonal parameters will be auto-tuned.

5.  Prophet workflow

“Prophet implements a procedure for forecasting time series data based
on an additive model where non-linear trends are fit with yearly,
weekly, and daily seasonality, plus holiday effects. It works best with
time series that have strong seasonal effects and several seasons of
historical data. Prophet is robust to missing data and shifts in the
trend, and typically handles outliers well.”

source: <https://cran.r-project.org/web/packages/prophet/index.html>

``` r
wfl_prophet <- workflow() %>%
  add_model(prophet_reg() %>%
              set_engine("prophet")) %>%
  add_recipe(recipe(elec_demand ~ ., extract_nested_train_split(nested_data_tbl)))
```

Now that we have all workflows, they can be fit by using
modeltime_nested_fit() function from modeltime package. The beauty of
this solution is its scalability. <br/> What it means? It means that
we’ll in fact build five different models per country. <br/> Before
that, good practice is to fit them on a single dataset (country) and
inspect for any potential errors (usually driven by data quality, namely
completeness). Let’s do it!<br/>

One of the argument of this function, important for our analysis is
conf_interval which can be interpret as range of possible outcomes for a
prediction calculated at certain probability. Default value of
probability is 0.95, let’s change it to 0.99.

To check for errors we can use below code.

``` r
try_sample_tbl %>%
  extract_nested_error_report()
```

    ## # A tibble: 0 x 4
    ## # ... with 4 variables: country <chr>, .model_id <int>, .model_desc <chr>,
    ## #   .error_desc <chr>

Our workflows are errors-free so let’s fit them on full dataset.<br/>
Fitting models can take some time. We can shorten it allowing parallel
run on PC’s cores. To do that (and first to check for number of cores,
functions from parallel library can be used - commented out in below
code). Alternatively we can set allow_par argument to TRUE will have
similar effect (although recommended is using parallel_start() function
with relevant number of cores.

``` r
parallel::detectCores()
```

    ## [1] 4

``` r
parallel_start(4)

nested_modeltime_tbl <- nested_data_tbl %>%
  modeltime_nested_fit(
    model_list = list(
      wflw_xgb,
      wflw_thief,
      wfl_exp_s,
      wfl_arima,
      wfl_prophet
    ),
    conf_interval = 0.99,
    control = control_nested_fit(
      verbose = FALSE,
      allow_par = FALSE
    )
  )
parallel_stop()
```

We haven’t specified any metrics in modeltime_nested_fit() function
hence default set of numeric metrics for regression models will be
provided. Customized list of metrics can be specified through metric_set
parameter (we’ll get back to this at later stage).<br/>

##### Inspection of models accuracy

Having fit models for all countries we can asses their accuracy using
available metrics. Evaluation is done on TEST set.
table_nested_test_accuracy() function creates nice html table with all
metrics per model per country (I’ll comment it out to simplify output
and the size of .md file)

``` r
nested_modeltime_tbl %>%
  extract_nested_test_accuracy() # %>%
```

    ## # A tibble: 60 x 10
    ##    country .model_id .model_desc  .type   mae  mape  mase smape  rmse   rsq
    ##    <chr>       <int> <chr>        <chr> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
    ##  1 Belgium         1 XGBOOST      Test  216.   2.72 0.606  2.77 294.  0.840
    ##  2 Belgium         2 TEMPORAL HI~ Test  166.   2.19 0.466  2.17 194.  0.933
    ##  3 Belgium         3 ETSMAA       Test  145.   1.91 0.409  1.90 181.  0.928
    ##  4 Belgium         4 ARIMA        Test  182.   2.41 0.512  2.38 220.  0.918
    ##  5 Belgium         5 PROPHET      Test  130.   1.71 0.367  1.71 164.  0.932
    ##  6 Denmark         1 XGBOOST      Test   71.4  2.22 0.354  2.26  96.2 0.945
    ##  7 Denmark         2 TEMPORAL HI~ Test   60.8  1.96 0.302  1.99  77.5 0.970
    ##  8 Denmark         3 ETSMADA      Test   63.0  2.02 0.313  2.06  79.5 0.970
    ##  9 Denmark         4 ARIMA        Test   88.0  2.75 0.436  2.81 108.  0.964
    ## 10 Denmark         5 PROPHET      Test   84.7  2.71 0.420  2.77 104.  0.963
    ## # ... with 50 more rows

``` r
  # table_modeltime_accuracy()
```

Forecast generated by all models can be visualized on test set using
below code. Plot generated by plot_modeltime_forecast function can be
interactive when ‘.interactive’ argument is set to TRUE. It’s a nice
feature as it allows zooming in.

``` r
nested_modeltime_tbl %>%
  extract_nested_test_forecast() %>%
  group_by(country) %>%
  plot_modeltime_forecast(.facet_ncol = 4, # when filter for country is removed
                          .conf_interval_show = FALSE,
                          .legend_show = TRUE,
                          .interactive = FALSE,
                          .title = 'Forecasts for test set')
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->
Plots are busy as all models are displayed, however at at this stage we
can also select best model per country. <br/> Default error ruducing
metric is ‘rmse’ but this can be also specified through ‘metric’
parameter inside modeltime_nested_select_best() function.

``` r
nested_best_tbl <- nested_modeltime_tbl %>%
  modeltime_nested_select_best() # try metric = "mae"
```

Forecast generated by best model per country can also be plotted.

``` r
nested_best_tbl %>%
  extract_nested_test_forecast() %>%
  # filter(country = 'Denmark') %>%
  group_by(country) %>%
  plot_modeltime_forecast(.facet_ncol = 4,
                          .conf_interval_show = TRUE,
                          .legend_show = TRUE,
                          .interactive = FALSE,
                          .title = 'Forecast from best model per country.'
                          )
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

Visual inspection of forecast for test set is important step in demand
forecasting. Equally important is inspection of forecast errors which is
critical input to Supply Planning for scenario planning and safety stock
calculation.<br/> In addition to that, we can also learn more about our
models and also reveal some weaknesses of a single error metric (however
good it is, like rmse) which can lead to further improvement or change
of demand forecasting method.<br/> As we’ll see shortly, it may require
building customized metrics (luckily, yardstick library provides all
necessary tools to do that).

We’ll use ggplot to explore forecast errors. Object nested_best_tbl
holds information about best models.

``` r
nested_best_tbl_extract <- nested_best_tbl %>%
    extract_nested_test_forecast()
```

With a little bit of data transformation we can get all we need to plot
errors and evaluate best models from that perspective. <br/>

At this point it’s important to highlight that error will be calculated
as prediction - actual (NOT actual - prediction). We’ll do it as we’ll
be investigating the presence of BIAS and it will be more intuitive
as:<br/> - consistent presence of positive errors will suggest presence
of positive BIAS,<br/> - consistent presence of negative errors will
suggest present of negative BIAS.

``` r
actual <- nested_best_tbl_extract %>%
  filter(.key == 'actual')

pred <- nested_best_tbl_extract %>%
  filter(.key == 'prediction')

vis_table <- actual %>%
  left_join(pred, 
            by = c('country', '.index'),
            keep = FALSE) %>%
  filter(.key.y == 'prediction') %>%
  rename(conf_low = .conf_lo.y,
         conf_high = .conf_hi.y,
         actual = .value.x,
         prediction = .value.y) %>%
  group_by(country) %>%
  mutate(error = prediction - actual,
         BIAS_cumsum = cumsum(prediction - actual)) %>%
  select(!c(.model_id.x, .model_desc.x, .conf_lo.x, .conf_hi.x)
         )
```

Let’s again look into best models, but this time we’ll only inspect test
set horizon which should greatly improve readability of our plot.

``` r
vis_table %>%
  filter(.key.y == 'prediction') %>%
  ggplot(group = country) +
  geom_line(aes(x = .index, y = actual)) +
  geom_line(aes(x = .index, y = prediction, color = .model_desc.y)) +
  geom_line(aes(x = .index, y = conf_low), 
            color = 'lightgrey') +
  geom_line(aes(x = .index, y = conf_high), 
            color = 'lightgrey') +
  theme_minimal() +
  facet_wrap(.~ country, ncol = 3, scales = 'free_y') +
  labs(color = 'model') +
  theme(legend.position = 'bottom', 
        plot.title = element_text(hjust = 0.5),
        axis.title.x = element_blank()) +
  ggtitle('Best models fit over test set horizon.')
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-27-1.png)<!-- -->

It makes a difference, doesn’t it? <br/> Colored line represent best
model based on rmse metric, lightgrey lines mark lower and upper
boundaries of confidence interval (95%). <br/>

**There are 2 interesting observations from above plots.**<br/>
**Firstly, best results (minimizing RMSE) are achieved through different
models, depending on a country. This shows that ‘one-fit all’ approach
doesn’t work in demand forecasting.**<br/> **Secondly, in some
countries, Machine Learning based models deliver best results,
confirming its value for demand forecasting.**

Finally, we can inspect forecast errors. First, let’s visualize their
distribution.

``` r
vis_table %>%
  group_by(country) %>%
  ggplot(aes(error, fill = .model_desc.y)) +
  geom_histogram() +
  theme_minimal() +
  facet_wrap(.~ country, 
             ncol = 3, 
             scales = 'free'
             ) +
  labs(fill = 'model') +
  theme(legend.position = 'bottom', 
        plot.title = element_text(hjust = 0.5),
        axis.title.x = element_blank()) +
  ggtitle('Best models: distribution of errors measured on test set horizon.')
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-28-1.png)<!-- -->

Indeed, distribution of errors in some countries is clearly skewed
either to left or right. Extreme example of this is Netherlands and
Denmark, which suggests presence of negative BIAS (distribution of error
calculated as prediction minus actual is mostly lower than 0 which means
underforecasting).<br/> This can be easier to notice using geom_density
instead of geom_histogram since we only have 24 discrete observations.
As a reminder, area under density curve is 1 (which is 100% of all
probabilities for value on x-axis).

``` r
vis_table %>%
  group_by(country) %>%
  ggplot(aes(error, fill = .model_desc.y)) +
  geom_density() +
  theme_minimal() +
  facet_wrap(.~ country, 
             ncol = 3, 
             scales = 'free'
             ) +
  labs(fill = 'model') +
  theme(legend.position = 'bottom',
        plot.title = element_text(hjust = 0.5),
        axis.title.x = element_blank()) +
  ggtitle('Best models: density of errors measured on test set horizon.')
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-29-1.png)<!-- -->

Let’s confirm our findings plotting BIAS (cumulative sum of error
calculated as prediction - actual) to check for any patterns of
CONSISTENT deviation from actual demand from test set.

``` r
vis_table %>%
  ggplot(group = country) +
  geom_point(aes(x = .index, y = BIAS_cumsum, color = .model_desc.y)) +
  geom_line(aes(x = .index, y = BIAS_cumsum, color = .model_desc.y)) +
  theme_minimal() +
  facet_wrap(.~ country, ncol = 3, scales = 'free_y') +
  labs(color = 'model') +
  theme(legend.position = 'bottom', 
        plot.title = element_text(hjust = 0.5),
        axis.title.x = element_blank()) +
  ggtitle('Best models: cumulative monthly errors over test set horizon.')
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-30-1.png)<!-- -->

We can indeed see clear BIAS in Netherlands and Denmark. It also appears
that most models are biased, however overall direction is often driven
by high error in one of months.<br/> For comparison, let’s look at
errors and if they oscillate around 0.

``` r
vis_table %>%
  ggplot(group = country) +
  geom_point(aes(x = .index, y = error, color = .model_desc.y)) +
  geom_line(aes(x = .index, y = error, color = .model_desc.y)) +
  geom_line(aes(x = .index, y = 0), color = 'black') +
  theme_minimal() +
  facet_wrap(.~ country, ncol = 3, scales = 'free_y') +
  labs(color = 'model') +
  theme(legend.position = 'bottom', 
        plot.title = element_text(hjust = 0.5),
        axis.title.x = element_blank()) +
  ggtitle('Best models: monthly errors over test set horizon.')
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-31-1.png)<!-- -->

Above plot confirms negative BIAS in Netherlands and Denmark and
presence of positive BIAS in UK and presence of negative BIAS in Greece
over majority of test horizon. <br/> Errors in other market oscillate
around 0, which indicates that there’s no major problem with BIAS.<br/>

Evaluation of models from BIAS perspective is not possible in modeltime
library simply because we’re restricted by available metrics.<br/> We
can however create custom metrics with a help of yardstick library which
provides a standard template for this. To present this functionality,
we’ll build a few simple BIAS-oriented metrics and accuracy-oriented
metrics to invest forecast precision.

We’ll create 5 metrics following same steps (described through
comments):<br/> Four of them will be BIAS-related, one focusing on
precision and in addition we’ll calculate standard error (se) which
could be used for evaluation of different boundaries of confidence
interval.

1.  tracking signal - sum of errors divided by mean absolute deviation
    (error); this is approach described in APICS CPIM; acceptable value
    depends on decision-makers.

``` r
# first step is to create vector version of function

track_sig_v <- function(truth, estimate, na_rm = TRUE, ...) {
  
  t_sig_impl <- function(truth, estimate) {
    sum(estimate - truth)/mean(abs(estimate - truth))
  }
  # next step is pass implementation of the function to metric_vec_template
  metric_vec_template(
    metric_impl = t_sig_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    ...
  )
}

# final step is data frame implementation of vector version function  


track_sig <- function(data, ...) {
  UseMethod("track_sig")
}
# direction can be set o 'zero', 'minimize' or 'maximize' which depends on nature
# of the metric
track_sig <- new_numeric_metric(track_sig, direction = "zero")

track_sig.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  metric_summarizer(
    metric_nm = "track_sig",
    metric_fn = track_sig_v,
    data = data,
    truth = !! enquo(truth),
    estimate = !! enquo(estimate),
    na_rm = na_rm,
    ...
  )
}
```

2.  BIAS_n - arithmetic sum of numeric representation of BIAS (-1 if
    negative BIAS, 0 if no BIAS, 1 if positive BIAS); value 0 suggests
    balanced model.

``` r
bias_n_v <- function(truth, estimate, na_rm = TRUE, ...) {
  
  bias_n_v_impl <- function(truth, estimate) {
    
    m <- estimate - truth
    m[m < 0] <- -1
    m[m == 0] <- 0
    m[m > 0] <- 1
    return(sum(m))
  }
  metric_vec_template(
    metric_impl = bias_n_v_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    ...
  )
}

bias_n <- function(data, ...) {
  UseMethod("bias_n")
}

bias_n <- new_numeric_metric(bias_n, direction = "zero")

bias_n.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  metric_summarizer(
    metric_nm = "bias_n",
    metric_fn = bias_n_v,
    data = data,
    truth = !! enquo(truth),
    estimate = !! enquo(estimate),
    na_rm = na_rm,
    ...
  )
}
```

3.  BIAS_c - cumulative sum of errors (the closer to 0 the highest
    alignment to total actual demand over test horizon).

``` r
bias_c_v <- function(truth, estimate, na_rm = TRUE, ...) {
  
  bias_c_v_impl <- function(truth, estimate) {
    
    m <- cumsum(estimate - truth)
    return(m[length(m)])
  }
  metric_vec_template(
    metric_impl = bias_c_v_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    ...
  )
}

bias_c <- function(data, ...) {
  UseMethod("bias_c")
}

bias_c <- new_numeric_metric(bias_c, direction = "zero")

bias_c.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  metric_summarizer(
    metric_nm = "bias_c",
    metric_fn = bias_c_v,
    data = data,
    truth = !! enquo(truth),
    estimate = !! enquo(estimate),
    na_rm = na_rm,
    ...
  )
}
```

4.  BIAS_m (arithmetic mean of errors)

``` r
bias_m_v <- function(truth, estimate, na_rm = TRUE, ...) {
  
  bias_m_v_impl <- function(truth, estimate) {
    
    mean(estimate - truth)
  }
  metric_vec_template(
    metric_impl = bias_m_v_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    ...
  )
}

bias_m <- function(data, ...) {
  UseMethod("bias_m")
}

bias_m <- new_numeric_metric(bias_m, direction = "zero")

bias_m.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  metric_summarizer(
    metric_nm = "bias_m",
    metric_fn = bias_m_v,
    data = data,
    truth = !! enquo(truth),
    estimate = !! enquo(estimate),
    na_rm = na_rm,
    ...
  )
}
```

5.  Accuracy - percentage of errors falling within +- 5% threshold<br/>

``` r
acc_95_v <- function(truth, estimate, na_rm = TRUE, ...) {

  acc_95_v_impl <- function(truth, estimate) {

    m <- estimate/truth
    m[m < 0.95 | m > 1.05] <- 0
    m[m >= 0.95 & m <= 1.05] <- 1
    return(sum(m)/length(m))
  }
  metric_vec_template(
    metric_impl = acc_95_v_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    ...
  )
}

acc_95 <- function(data, ...) {
  UseMethod("acc_95")
}

acc_95 <- new_numeric_metric(acc_95, direction = "maximize")

acc_95.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  metric_summarizer(
    metric_nm = "acc_95",
    metric_fn = acc_95_v,
    data = data,
    truth = !! enquo(truth),
    estimate = !! enquo(estimate),
    na_rm = na_rm,
    ...
  )
}
```

6.  standard error of forecast - used to calculate confidence intervals
    at given alpha; we’ll use it later for safety stock modelling.
    Please note that although ‘truth’ argument is not used in metric
    calculation, it still needs to be passed to the function in order to
    use this framework.

``` r
se_v <- function(truth, estimate, na_rm = TRUE, ...) {

  se_v_impl <- function(truth, estimate) {

    m <- sd(estimate)
    n <- length(estimate)
    se <- m/sqrt(n)
    return(se)
  }
  metric_vec_template(
    metric_impl = se_v_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    ...
  )
}

se <- function(data, ...) {
  UseMethod("se")
}

se <- new_numeric_metric(se, direction = "minimize")

se.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  metric_summarizer(
    metric_nm = "se",
    metric_fn = se_v,
    data = data,
    truth = !! enquo(truth),
    estimate = !! enquo(estimate),
    na_rm = na_rm,
    ...
  )
}
```

Such created custom metrics can be passed to an object (together with
rmse and mea which will be used later for safety stocks modelling)…

``` r
cust_metrics <- metric_set(rmse, mae, track_sig, bias_n, bias_c, bias_m, acc_95, se)
```

…and further to metric_set argument inside modeltime_nested_fit
function.<br/> We’ll re-do modelling only for countries with presence of
BIAS.

``` r
check_bias_nested_tbl <- nested_data_tbl %>%
  dplyr::filter(country %in% c('Netherlands', 'Greece', 'Denmark', 'UK')) %>%
  modeltime_nested_fit(
    model_list = list(
      wflw_xgb,
      wflw_thief,
      wfl_exp_s,
      wfl_arima,
      wfl_prophet
    ),
    metric_set = cust_metrics, # custom metrics
    conf_interval = 0.99,
    control = control_nested_fit(
      verbose = TRUE,
      allow_par = FALSE
    )
  )
```

Let’s once more time visualize models for the four countries. Again,
we’ll do that only for test horizon to maximize readability of our
plots.<br/> We’ll create separate plot for each country and inspect
models visually cross-checking with results of our custom metrics (and
rmse) in order to pick best model.

``` r
check_bias_nested_tbl %>%
  extract_nested_test_forecast() %>%
  filter(country == "Netherlands") %>%
  filter(.index >= as.Date('2002-06-01')) %>%
  ggplot() +
  geom_line(aes(x = .index, y = .value, color = .model_desc), size = 1.5) +
  theme_minimal() +
  labs(color = 'model') +
  theme(legend.position = 'bottom', 
        plot.title = element_text(hjust = 0.5), 
        axis.title.x = element_blank()
        ) +
  ggtitle('Netherlands: Models fit over test set horizon.') +
  scale_color_brewer(palette = 'RdGy')
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-40-1.png)<!-- -->
Netherlands: custom error metrics’ results.

``` r
check_bias_nested_tbl %>%
  extract_nested_test_accuracy() %>%
  filter(country == 'Netherlands') %>%
  select(-c(country, .model_id, .type))
```

    ## # A tibble: 5 x 9
    ##   .model_desc        rmse   mae track_sig bias_n bias_c bias_m acc_95    se
    ##   <chr>             <dbl> <dbl>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl> <dbl>
    ## 1 XGBOOST            200.  140.     -24.0    -22 -3352.  -140.  1      86.2
    ## 2 TEMPORAL HIERARC~  228.  198.      18.9     16  3730.   155.  1      89.5
    ## 3 ETSMAA             239.  206.      18.2     16  3757.   157.  1      85.3
    ## 4 ARIMA              237.  203.      19.7     16  4000.   167.  1      89.5
    ## 5 PROPHET            314.  270.      21.7     18  5849.   244.  0.833  86.4

``` r
check_bias_nested_tbl %>%
  extract_nested_test_forecast() %>%
  filter(country == "Greece") %>%
  filter(.index >= as.Date('2002-06-01')) %>%
  ggplot() +
  geom_line(aes(x = .index, y = .value, color = .model_desc), size = 1.5) +
  theme_minimal() +
  labs(color = 'model') +
  theme(legend.position = 'bottom', 
        plot.title = element_text(hjust = 0.5), 
        axis.title.x = element_blank()
        ) +
  ggtitle('Greece: Models fit over test set horizon.') +
  scale_color_brewer(palette = 'RdGy')
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-42-1.png)<!-- -->

Greece: custom error metrics’ results.

``` r
check_bias_nested_tbl %>%
  extract_nested_test_accuracy() %>%
  filter(country == 'Greece') %>%
  select(-c(country, .model_id, .type))
```

    ## # A tibble: 5 x 9
    ##   .model_desc        rmse   mae track_sig bias_n bias_c bias_m acc_95    se
    ##   <chr>             <dbl> <dbl>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl> <dbl>
    ## 1 XGBOOST            223.  165.     -12.0     -2 -1992.  -83.0  0.75   75.2
    ## 2 TEMPORAL HIERARC~  268.  227.      22.3     20  5072.  211.   0.542  77.9
    ## 3 ETSMADM            188.  137.     -15.4     -8 -2105.  -87.7  0.792  65.5
    ## 4 ARIMA              276.  235.      20.8     18  4887.  204.   0.583  77.1
    ## 5 PROPHET            310.  258.      15.2     14  3911.  163.   0.542  50.7

``` r
check_bias_nested_tbl %>%
  extract_nested_test_forecast() %>%
  filter(country == "Denmark") %>%
  filter(.index >= as.Date('2002-06-01')) %>%
  ggplot() +
  geom_line(aes(x = .index, y = .value, color = .model_desc), size = 1.5) +
  theme_minimal() +
  labs(color = 'model') +
  theme(legend.position = 'bottom', 
        plot.title = element_text(hjust = 0.5), 
        axis.title.x = element_blank()
        ) +
  ggtitle('Denmark: Models fit over test set horizon.') +
  scale_color_brewer(palette = 'RdGy')
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-44-1.png)<!-- -->
Denmark: Custom error metrics’ results.

``` r
check_bias_nested_tbl %>%
  extract_nested_test_accuracy() %>%
  filter(country == 'Denmark') %>%
  select(-c(country, .model_id, .type))
```

    ## # A tibble: 5 x 9
    ##   .model_desc        rmse   mae track_sig bias_n bias_c bias_m acc_95    se
    ##   <chr>             <dbl> <dbl>     <dbl>  <dbl>  <dbl>  <dbl>  <dbl> <dbl>
    ## 1 XGBOOST            96.0  71.8     -17.7     -6 -1271.  -53.0  0.875  62.7
    ## 2 TEMPORAL HIERARC~  77.5  60.8     -20.9    -18 -1271.  -53.0  0.958  68.0
    ## 3 ETSMADA            79.5  63.0     -21.0    -18 -1327.  -55.3  0.958  67.6
    ## 4 ARIMA             108.   88.0     -23.4    -20 -2057.  -85.7  0.875  62.5
    ## 5 PROPHET           104.   84.7     -23.4    -20 -1987.  -82.8  0.958  66.6

``` r
check_bias_nested_tbl %>%
  extract_nested_test_forecast() %>%
  filter(country == "UK") %>%
  filter(.index >= as.Date('2002-06-01')) %>%
  ggplot() +
  geom_line(aes(x = .index, y = .value, color = .model_desc), size = 1.5) +
  theme_minimal() +
  labs(color = 'model') +
  theme(legend.position = 'bottom', 
        plot.title = element_text(hjust = 0.5), 
        axis.title.x = element_blank()
        ) +
  ggtitle('UK: Models fit over test set horizon.') +
  scale_color_brewer(palette = 'RdGy')
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-46-1.png)<!-- -->
UK: custom error metrics’ results.

``` r
check_bias_nested_tbl %>%
  extract_nested_test_accuracy() %>%
  filter(country == 'UK') %>%
  select(-c(country, .model_id, .type))
```

    ## # A tibble: 5 x 9
    ##   .model_desc       rmse   mae track_sig bias_n  bias_c bias_m acc_95    se
    ##   <chr>            <dbl> <dbl>     <dbl>  <dbl>   <dbl>  <dbl>  <dbl> <dbl>
    ## 1 XGBOOST          2012. 1380.    -16.0      -6 -22092.  -920.  0.792  693.
    ## 2 TEMPORAL HIERAR~ 1419. 1056.     -5.90     -2  -6227.  -259.  0.708  883.
    ## 3 ETSMAA           1387. 1037.     -4.05      0  -4203.  -175.  0.75   884.
    ## 4 ARIMA            1486. 1156.      5.16      8   5964.   249.  0.75   733.
    ## 5 PROPHET          1353. 1097.      7.63     10   8365.   349.  0.792  879.

Additional metrics shed more light on the problem of BIAS and also on
precision of our models. Results of metrics indicate that other ‘best’
models could be chosen if rmse was not a primary error criterion.<br/>

As for our analysis, let’s stick to rmse taking into account high
precision of models. The problem of BIAS was mainly meant to show
flexibility in terms of development of custom metrics.

##### Refitting models and forecasting future months.

Best models can now be refit on full dataset (train + test) using
modeltime_nested_refit function from modeltime library. Parallel
processing should kick in automatically, but it is recommended to set up
a Parallel Backend with parallel_start() function.

``` r
nested_best_refit_tbl <- nested_best_tbl %>%
  modeltime_nested_refit(
    control = control_refit(
      verbose = TRUE,
      allow_par = TRUE
    )
  )
```

Refit models can be now used to forecast future months (outside of test
set horizon) set at the beginning of analysis (12 months) using function
extract_nested_future_forecast.<br/>

Forecast naturally can also be visualized. We’ll used
plot_modeltime_forecast from modeltime package, alternatively for more
elegant results ggplot could be used. This would require small data
manipulation first so we’ll stick to tools (plot) provided by modeltime.

``` r
nested_best_refit_tbl %>%
  extract_nested_future_forecast() %>%
  group_by(country) %>%
  plot_modeltime_forecast(.facet_ncol = 4,
                          .conf_interval_show = TRUE,
                          .legend_show = TRUE,
                          .interactive = FALSE,
                          .title = 'Forecast for future months per country.'
                          )
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-49-1.png)<!-- -->

Quick visual inspection tells that forecast for most countries looks
fine. Exception is Greece where it doesn’t follow upward trend. This can
suggest that Exponential Smoothing method should be changed for other
model - one with similar precision and less evident negative BIAS
(XGBoost) maybe?<br/>

Modeltime library provides yet another useful function -
modeltime_nested_forecast, which allows quick change of forecast horizon
with argument ‘h’. In below example, we change forecast horizon from 12
to 18 months.

``` r
new_forecast_tbl <- nested_best_refit_tbl %>%
  modeltime_nested_forecast(
    h = 18,
    conf_interval = 0.99,
    control = control_nested_forecast(
      verbose = FALSE,
      allow_par = TRUE
      )
  )
```

##### Safety stock modelling using expected forecast errors

<br/>

To model safety stock as buffer for demand volatility, invaluable
information is expected forecast error.<br/> Contrary to BIAS-related
topics, error will be calculated as actual - estimate (forecast). This
is important distinction focuses on the risk to customer service coming
from underforecasting (any time actual is higher than prediction).

There’s multiple ways to calculate safety stocks to protect against
demand inaccuracy, mostly related to selection of error metric and
calculation of safety factor (difference comes from the type of
distribution of the error). <br/> We’ll investigate a couple of
scenarios aiming to achieve 99% of customer service level and to
simplify, we’ll assume normal distribution of the error:<br/> 1. rmse as
error metric, 2. mea as error metric, 3. use the difference between
upper level of confidence interval and forecast as safety stock level.

Let’s first visualize forecast errors.

We can use previously created vis_table object filtered for countries in
scope. In this step we can also calculate the difference between upper
value of confidence interval and forecast.

``` r
four_countries <- vis_table %>%
  dplyr::filter(country %in% c('Netherlands', 'Greece', 'Denmark', 'UK')) %>%
  mutate(error = actual - prediction,  # overwrite error 
         safety_stock_ci = conf_high - prediction) # diff between conf_high and forecast
  

four_countries %>%
  ggplot(group = country) +
  geom_point(aes(x = .index, y = error, color = .model_desc.y)) +
  geom_line(aes(x = .index, y = error, color = .model_desc.y)) +
  theme_minimal() +
  facet_wrap(.~ country, ncol = 2, scales = 'free_y') +
  labs(color = 'model') +
  theme(legend.position = 'bottom', 
        plot.title = element_text(hjust = 0.5),
        axis.title.x = element_blank()) +
  ggtitle('Best models: monthly errors over test set horizon.')
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-51-1.png)<!-- -->
rmse and mae can be extracted from models error’s metrics (we’ll pick
lowest rmse from each country).<br/> For calculation of safety factor,
we’ll use qnorm function providing required service level of 99%.
Required safety stock will be calculated by multiplying both
values.<br/> This will require some data manipulation.<br/>

``` r
alpha = 0.01

(safety_stocks <- check_bias_nested_tbl %>%
   extract_nested_test_accuracy() %>%
   select(country, .model_desc, rmse, mae) %>%
   group_by(country) %>%
   mutate(min_rmse = min(rmse)) %>%
   filter(rmse == min_rmse) %>%
   mutate(safety_factor = qnorm(1-alpha), # normal_distr
          safety_stock_rmse = rmse * safety_factor,
          safety_stock_mae = mae * safety_factor,
          ) %>%
   select(-c(min_rmse, .model_desc))
)
```

    ## # A tibble: 4 x 6
    ## # Groups:   country [4]
    ##   country       rmse    mae safety_factor safety_stock_rm~ safety_stock_mae
    ##   <chr>        <dbl>  <dbl>         <dbl>            <dbl>            <dbl>
    ## 1 Denmark       77.5   60.8          2.33             180.             141.
    ## 2 Greece       188.   137.           2.33             437.             319.
    ## 3 Netherlands  200.   140.           2.33             465.             325.
    ## 4 UK          1353.  1097.           2.33            3148.            2551.

Let’s add safety stocks to table with errors and visualize it.

``` r
four_countries_all <- four_countries %>%
    left_join(safety_stocks,
            by = c('country'),
            keep = FALSE)

four_countries_all %>%
  ggplot() +
  geom_point(aes(x = .index, y = error, color = .model_desc.y)) +
  geom_line(aes(x = .index, y = error, color = .model_desc.y)) +
  geom_line(aes(x = .index, y = safety_stock_rmse), 
            color = 'black', 
            show.legend = TRUE, 
            linetype = 'dashed') +
  geom_line(aes(x = .index, y = safety_stock_mae), 
            color = 'red',
            linetype = 'dashed') +
  geom_line(aes(x = .index, y = safety_stock_ci), 
            color = 'blue',
            linetype = 'dashed') +
  theme_minimal() +
  facet_wrap(.~ country, ncol = 2, scales = 'free_y') +
  labs(color = 'model') +
  theme(legend.position = 'bottom', 
        plot.title = element_text(hjust = 0.5),
        axis.title.x = element_blank()) +
  ggtitle('Best models: monthly errors and rmse-based safety stocks.')
```

![](Scalable_Demand_Forecasting_files/figure-gfm/unnamed-chunk-53-1.png)<!-- -->
Black horizontal line represents rmse-based safety stocks, red line is
mae-based safety stock and blue line is ci-based safety stock. mae
provides lowest safety stock, which is in line with nature of this
metric. It’s not outliers sensitive like rmse, which penalizes high
errors. ci-based safety stock seems to provide full protection against
underforecasting.<br/>

Let’s calculate actual service level over test horizon for each safety
stock level.<br/> For each level of safety stocks we’ll first calculate
portion of monthly actual demand above it and compare it with total
actual demand to calculate % of fill rate.

``` r
four_countries_all %>%
  select(actual, country, error, safety_stock_rmse, safety_stock_mae, safety_stock_ci) %>%
  mutate(
    uncovered_rmse = if_else(safety_stock_rmse - error < 0, 
                            abs(safety_stock_rmse - error),
                            0),
    uncovered_mae = if_else(safety_stock_mae - error < 0, 
                            abs(safety_stock_mae - error),
                            0),
    uncovered_ci = if_else(safety_stock_ci - error < 0, 
                            abs(safety_stock_ci - error),
                            0)
  ) %>%
  group_by(country) %>%
  summarise(total_actual = sum(actual),
         total_uncovered_rmse = sum(uncovered_rmse),
         total_uncovered_mae = sum(uncovered_mae),
         total_uncovered_ci = sum(uncovered_ci),
         service_level_rmse = (total_actual - total_uncovered_rmse)/total_actual,
         service_level_mae = (total_actual - total_uncovered_mae)/total_actual,
         service_level_ci = (total_actual - total_uncovered_ci)/total_actual) %>%
  select(country, service_level_rmse, service_level_mae, service_level_ci) %>%
  mutate_at(2:4, round, 4)
```

    ## # A tibble: 4 x 4
    ##   country     service_level_rmse service_level_mae service_level_ci
    ##   <chr>                    <dbl>             <dbl>            <dbl>
    ## 1 Denmark                   1.00             0.999                1
    ## 2 Greece                    1.00             0.998                1
    ## 3 Netherlands               1.00             0.998                1
    ## 4 UK                        1                1.00                 1

With quick glance at above table we can tell that all safety stock
levels provide required (or actually higher) service level (here
interpreted as fill rate). <br/> **In this case we can select mae-based
safety stocks, which will be the least costly option.**<br/>

### SUMMARY

<br/>

As we’ve seen through the course of this project, R provides all
necessary tools to build advanced and scalable demand forecasting
models, both based on classic time series analysis and machine learning
algorithms.<br/>

In this project we’ve:<br/> 1. described standard approach to demand
forecasting using train, test and future datasets,<br/> 2. built five
different models per country,<br/> 3. used one of provided numeric error
metric to select model that minimizes the error,<br/> 4. showed how to
create customized metrics that focuses on BIAS and precision of
forecast,<br/> 5. forecasted future demand and showed how to change
demand horizon if needed,<br/> 6. on the basis of expected error,
explored different ways to calculate safety stocks to achieve required
service level,<br/> 7. provided a method to select preferred safety
stock.
