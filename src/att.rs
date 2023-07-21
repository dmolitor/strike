use polars::frame::UniqueKeepStrategy;
use polars::prelude::{DataFrame, DataFrameJoinOps, PolarsResult};

use crate::distance::nn_match;

// Calculate the ATT
//
// This function takes two DataFrames. The first is the full treated sample
// and the second is the matched control sample. Then, given the name of
// the outcomes column, it calculates and returns the ATT. The ATT is
// simply the mean of the element-wise difference between the treated
// outcomes and control outcomes.
// NOTE: This is NOT the bias-corrected ATT estimator.
pub fn calculate_att(
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

// Calculate the ATT variance
//
// This function implements the consistent estimator for the variance of the
// matching estimator for the ATT, as proposed in 
// "Large Sample Properties of Matching Estimators for Average Treatment 
// Effects" (Abadie and Imbens, 2006).
// NOTE: This is the variance estimation that allows for heteroskedasticity.
pub fn calculate_variance(
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

// Estimates the observation-level conditional variance as a necessary piece of
// estimating the full-sample ATT variance.
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

// Estimates the number of times each observation is used as a match
// since we are doing matching with replacement. A necessary piece of
// estimating the full-sample ATT variance.
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