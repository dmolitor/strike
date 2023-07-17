use polars::prelude::*;
use polars::datatypes::DataType::{Float64, Int64};
use linfa::dataset::Dataset;
use linfa::traits::{Fit, Predict, Transformer};
use linfa_logistic::LogisticRegression;
use linfa_logistic::error::{Result as LinfaResult};
use linfa_preprocessing::linear_scaling::LinearScaler;
use ndarray::{Array1, Array2, Axis};
use std::error::Error;

fn import_data(path: &str) -> PolarsResult<DataFrame> {
    let data = CsvReader::from_path(path)?.finish()?;
    Ok(data)
}

fn estimate_logit(
    x: Array2<f64>,
    d: Array1<i64>,
    feat_names: Vec<&str>
) -> Result<Array1<f64>, Box<dyn Error>> {
    let train = Dataset::new(x.clone(), d).with_feature_names(feat_names);
    //println!("#targets: {}\nFeature names: {:?}\n", &prop_df.ntargets(), &prop_df.feature_names());
    let scaler = LinearScaler::standard().fit(&train)?;
    let train = scaler.transform(train);
    let x = scaler.transform(x);
    let model = LogisticRegression::default()
        .with_intercept(true)
        .alpha(0.0)
        .fit(&train)?;
    let propensities = model.predict_probabilities(&x);
    //println!("Model view: {:?}\nTargets: {:?}\nPropensities: {:?}\n", &model, &train.targets(), &propensities);
    Ok(propensities)
}

/*fn nn_match(data: &DataFrame) -> PolarsResult<DataFrame> {
    let propensities = data
        .column(&"propensities")
        .expect("This column should never be missing.");
    for pscore in propensities {

    }
}*/

fn find_nn(data: &DataFrame, pscore: f64) -> PolarsResult<DataFrame> {
    let propensity_diff = data.column(&"propensities")? - pscore;
    let min_propensity_diff: Option<f64> = propensity_diff.abs()?.min();
    //data.filter(propensity_diff.eq(min_propensity_diff))
    match min_propensity_diff {
        Some(min_pscore) => {
            let propensity_mask = propensity_diff.equal(min_pscore)?;
            return Ok(data.filter(&propensity_mask)?.head(Some(1)));
        },
        None => panic!("find_nn: No nearest neighbor propensity score could be calculated")
    }
}

fn estimate_propensities<'a>(
    data: &'a mut DataFrame,
    treatment: &str
) -> Result<&'a mut DataFrame, Box<dyn Error>> {
    let (x, treat, feat_names) = construct(data, treatment)?;
    let propensities = Series::new(
        "propensities",
        estimate_logit(x, treat, feat_names)?.to_vec()
    );
    data.with_column(propensities)?;
    Ok(data)
}

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

fn treat_control_split(data: &DataFrame, treatment: &str) -> PolarsResult<(DataFrame, DataFrame)> {
    let mask_treat = data.column(treatment)?.equal(1)?;
    let mask_control = data.column(treatment)?.equal(0)?;
    let treat = data.filter(&mask_treat)?;
    let control = data.filter(&mask_control)?;
    Ok((treat, control))
}

fn main() -> Result<(), Box<dyn Error>> {
    // Import data
    let mut ecls = import_data(&"data/ecls-analysis.csv")?;
    // Estimate propensity score with Logistic regression
    let ecls = estimate_propensities(&mut ecls, &"catholic")?;
    let (ecls_treat, ecls_control) = treat_control_split(&ecls, &"catholic")?;
    let ecls_control = find_nn(&ecls_control, 0.494046);
    println!("Treated:\n{:?}\n----------------------\nControl:\n{:?}", &ecls_treat, &ecls_control);
    Ok(())
}
