use linfa::dataset::Dataset;
use linfa::traits::{Fit}; //, Transformer};
use linfa_logistic::LogisticRegression;
// use linfa_preprocessing::linear_scaling::LinearScaler;
use ndarray::{Array1, Array2};
use polars::datatypes::DataType::Int64;
use polars::prelude::{DataFrame, Float64Type, NamedFrom, PolarsResult, Series};
use std::error::Error;

// Prep a DataFrame for logistic regression with Linfa
//
// Given a Polars DataFrame and a string specifying a binary treatment variable
// this function returns a tuple containing predictors as a 2D ndarray, binary
// response as a 1D ndarray, and predictor names as a Vec.
fn construct<'a>(
    data: &'a DataFrame,
    treatment: &'a str
) -> PolarsResult<(Array2<f64>, Array1<i64>, Vec<&'a str>)> {
    let d = data.column(treatment)?
        .cast(&Int64)?
        .i64()?
        .to_ndarray()?
        .to_owned();
    let x = data.to_ndarray::<Float64Type>()?;
    let feat_names = data.get_column_names();
    Ok((x, d, feat_names))
}

// Estimate logistic regression with Linfa
//
// This function takes the output of `construct` and creates a Linfa Dataset.
// A logistic regression is fit on the full dataset. The estimated
// propensities are simply the fitted values, and are returned as a 1D ndarray.
fn estimate_logit(
    x: Array2<f64>,
    d: Array1<i64>,
    feat_names: Vec<&str>
) -> Result<Array1<f64>, Box<dyn Error>> {
    let train = Dataset::new(x.clone(), d).with_feature_names(feat_names);
    // The lines below normalize predictors but this seems to cause severe
    // overfitting so am dropping for now.
    // ===================================================
    // let scaler = LinearScaler::standard().fit(&train)?;
    // let train = scaler.transform(train);
    // let x = scaler.transform(x);
    let model = LogisticRegression::default()
        .with_intercept(true)
        .alpha(0.0)
        .fit(&train)?;
    let propensities = model.predict_probabilities(&x);
    Ok(propensities)
}

// Estimate propensity scores
//
// This function pulls all the propensity score estimation steps together.
// Given a DataFrame and the treatment column name it estimates
// propensity scores and appends them to the original DataFrame. It also
// appends a unique id to each observation, which is essential for
// downstream processing.
pub fn estimate_propensities<'a>(
    data: &'a mut DataFrame,
    treatment: &str
) -> Result<&'a mut DataFrame, Box<dyn Error>> {
    let (x, treat, feat_names) = construct(data, treatment)?;
    let propensities = Series::new(
        "propensities",
        estimate_logit(x, treat, feat_names)?.to_vec()
    );
    data.with_column(propensities)?;
    let idx:Vec<i64> = (1..=data.height() as i64).collect();
    let ids = Series::new("strike_id", idx);
    data.with_column(ids)?;
    Ok(data)
}