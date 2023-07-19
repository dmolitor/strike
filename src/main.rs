use polars::prelude::*;
use polars::datatypes::DataType::Int64;
use polars::frame::UniqueKeepStrategy;
use linfa::dataset::Dataset;
use linfa::traits::{Fit, Transformer};
use linfa_logistic::LogisticRegression;
use linfa_preprocessing::linear_scaling::LinearScaler;
use ndarray::{Array1, Array2};
use std::error::Error;
use std::iter::zip;

fn import_data(path: &str) -> PolarsResult<DataFrame> {
    let data = CsvReader::from_path(path)?.finish()?;
    Ok(data)
}

fn calculate_att(
    treat: &DataFrame,
    control: &DataFrame,
    outcome: &str
) -> PolarsResult<f64> {
    let observed_y = treat.column(outcome)?;
    let matched_y = control.column(outcome)?;
    let y_diff = (observed_y - matched_y).clone();
    let att: Vec<Option<f64>> = y_diff.mean_as_series().f64()?.to_vec();
    let att: &Option<f64> = att.first().unwrap();
    match att {
        Some(a) => {
            return Ok(*a);
        },
        None => panic!("calculate_att: ATT could not be calculated!")
    }
}

fn calculate_variance(
    treat: &DataFrame,
    control: &DataFrame,
    outcome: &str,
    treatment: &str
) -> PolarsResult<f64> {
    let treat_with_variance = subsample_conditional_variance(treat, outcome)?;
    let control_with_variance = subsample_conditional_variance(control, outcome)?;
    let treat_control = treat_with_variance.vstack(&control_with_variance)?;
    let sample_treat = treat_control.column(treatment)?;
    let sample_id_count = treat_control
        .column("strike_id_count")
        .expect("This column should never be missing!");
    let sample_conditional_var = treat_control
        .column("conditional_variance")
        .expect("This column should never be missing!");
    let n_treat_sq = (treat.height() * treat.height()) as f64;
    let weighted_treat = sample_treat - &(((sample_treat - 1) * -1) * sample_id_count.clone());
    let weighted_treat = &weighted_treat * &weighted_treat;
    let treat_by_var: Option<f64> = (weighted_treat * sample_conditional_var.clone()).sum();
    match treat_by_var {
        Some(t) => {
            let att_variance = t / n_treat_sq;
            return Ok(att_variance);
        },
        None => panic!("calculate_variance: Failed to calculate corrected ATT variance")
    }
}

fn estimate_logit(
    x: Array2<f64>,
    d: Array1<i64>,
    feat_names: Vec<&str>
) -> Result<Array1<f64>, Box<dyn Error>> {
    let train = Dataset::new(x.clone(), d).with_feature_names(feat_names);
    let scaler = LinearScaler::standard().fit(&train)?;
    let train = scaler.transform(train);
    let x = scaler.transform(x);
    let model = LogisticRegression::default()
        .with_intercept(true)
        .alpha(0.0)
        .fit(&train)?;
    let propensities = model.predict_probabilities(&x);
    Ok(propensities)
}

fn nn_match(main: &DataFrame, target: &mut DataFrame) -> PolarsResult<DataFrame> {
    let propensities = main
        .column(&"propensities")
        .expect("This column should never be missing!")
        .f64()?
        .to_vec();
    let ids = main
        .column(&"strike_id")
        .expect("This column should never be missing!")
        .i64()?
        .to_vec();
    let mut targets = DataFrame::empty();
    for (pscore, strike_id) in zip(propensities.iter(), ids.iter()) {
        let pscore = pscore.unwrap();
        let strike_id = strike_id.unwrap();
        let nearest_neighbor = find_nn(target, pscore, strike_id)?;
        targets = targets.vstack(&nearest_neighbor)?;
    }
    Ok(targets)
}

fn find_nn(data: &DataFrame, pscore: f64, strike_id: i64) -> PolarsResult<DataFrame> {
    let id_mask = data
        .column(&"strike_id")
        .expect("This column should never be missing!")
        .not_equal(strike_id)?;
    let data = data.filter(&id_mask)?;
    let pscore_diff = data
        .column(&"propensities")
        .expect("This column should never be missing!") 
        - pscore;
    let min_pscore_diff: Option<f64> = pscore_diff.abs()?.min();
    match min_pscore_diff {
        Some(p) => {
            let pscore_mask = pscore_diff.abs()?.equal(p)?;
            if !pscore_mask.any() {
                panic!("find_nn: No nearest neighbor control observation was found");
            }
            return Ok(data.filter(&pscore_mask)?.head(Some(1)));
        },
        None => panic!("find_nn: Failed to calculate propensity score distances")
    }
}

fn subsample_conditional_variance(
    data: &DataFrame,
    outcome: &str
) -> PolarsResult<DataFrame> {
    let mut data = subsample_count_matches(data)?;
    let self_matches = nn_match(&data, &mut data.clone())?;
    let observed_y = data.column(outcome)?;
    let matched_y = self_matches.column(outcome)?;
    let mean_y = (observed_y + matched_y) / 2.0;
    let mut cond_var = ((observed_y - &mean_y) * (observed_y - &mean_y))
        + ((matched_y - &mean_y) * (matched_y - &mean_y));
    let cond_var = cond_var.rename("conditional_variance").clone();
    data.with_column(cond_var)?;
    Ok(data)
}

fn subsample_count_matches(data: &DataFrame) -> PolarsResult<DataFrame> {
    let counts = data
        .groupby(["strike_id"])?
        .select(["strike_id"])
        .count()?;
    let data = data
        .left_join(&counts, ["strike_id"], ["strike_id"])?
        .unique(None, UniqueKeepStrategy::First, None)?;
    Ok(data)
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
    let idx:Vec<i64> = (1..=data.height() as i64).collect();
    let ids = Series::new("strike_id", idx);
    data.with_column(ids)?;
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

fn matches(
    data: &DataFrame,
    treatment: &str,
    outcome: &str
) -> Result<(f64, f64, DataFrame, DataFrame), Box<dyn Error>> {
    let mut strike = data.clone();
    let strike = estimate_propensities(&mut strike, treatment)?;
    let (strike_treat, mut strike_control) = treat_control_split(&strike, treatment)?;
    let strike_control = nn_match(&strike_treat, &mut strike_control)?;
    let att = calculate_att(&strike_treat, &strike_control, outcome)?;
    let att_variance = calculate_variance(&strike_treat, &strike_control, outcome, treatment)?;
    Ok((att, att_variance, strike_treat, strike_control))
}

fn main() {
    // Import data
    let ecls = import_data(&"data/ecls-analysis.csv").unwrap();
    // Estimate ATT and Variance with 1:1 matching on propensity score with replacement
    let (att, att_variance, _treat, _control) = matches(&ecls, &"catholic", &"c5r2mtsc_std").unwrap();
    let conf_int = (att - 1.96 * att_variance.sqrt(), att + 1.96 * att_variance.sqrt());
    // Display the ATT and corresponding 95% confidence interval
    println!("ATT: {:.3} | 95% Conf. Interval: ({:.3}, {:.3})", att, conf_int.0, conf_int.1);
    ()
}
