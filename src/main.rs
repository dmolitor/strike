mod att;
mod distance;
mod propensity;

use polars::frame::UniqueKeepStrategy;
use polars::prelude::{
    ChunkCompare,
    CsvReader,
    DataFrame,
    PolarsResult,
    SerReader
};
use std::env;
use std::error::Error;
use std::fmt;

use crate::att::{calculate_att, calculate_variance};
use crate::distance::nn_match;
use crate::propensity::estimate_propensities;

// Simple class containing the results from an estimated 1:1 propensity
// score matching routine.
#[derive(Debug)]
struct Strike {
    att: f64,
    att_variance: f64,
    treat: DataFrame,
    control: DataFrame
}

// Implement `Display` for `Strike`.
impl fmt::Display for Strike {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (lb, ub) = (
            self.att - 1.96 * self.att_variance.sqrt(),
            self.att + 1.96 * self.att_variance.sqrt()
        );
        let (n_treat, n_control) = n_treat_control(&self.treat, &self.control).unwrap();
        write!(
            f,
            "STRIKE =======================================\n\n\
            # Treat: {} | # Control (distinct): {}\n\n\
            ATT                     : {:.3}\n\
            Variance                : {:.3}\n\
            95% Confidence Interval : ({:.3}, {:.3})\n",
            n_treat,
            n_control,
            self.att,
            self.att_variance,
            lb,
            ub
        )
    }
}

// Imports a csv file from a specified path to a Polars DataFrame
fn import_data(path: &str) -> PolarsResult<DataFrame> {
    let data = CsvReader::from_path(path)?.finish()?;
    Ok(data)
}

// Count the number of unique treatment and control observations
fn n_treat_control(treat: &DataFrame, control: &DataFrame) -> PolarsResult<(i64, i64)> {
    let n_treat = treat.height() as i64;
    let n_control = control
        .unique(None, UniqueKeepStrategy::First, None)?
        .height() as i64;
    Ok((n_treat, n_control))
}

// Estimate ATT with 1:1 propensity score matching
//
// This is the whole thing. Estimate propensities, perform 1:1 matching on
// the propensity scores with replacement, and calculate the ATT and
// variance.
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

// Split a DataFrame into treatment and control sub-frames.
fn treat_control_split(data: &DataFrame, treatment: &str) -> PolarsResult<(DataFrame, DataFrame)> {
    let mask_treat = data.column(treatment)?.equal(1)?;
    let mask_control = data.column(treatment)?.equal(0)?;
    let treat = data.filter(&mask_treat)?;
    let control = data.filter(&mask_control)?;
    Ok((treat, control))
}

fn main() {
    // Import command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        panic!("Expected 3 arguments but {} {:?} were provided", (&args.len() - 1), &args[1..].to_vec());
    }

    // Extract args to necessary variables
    let path = &args[1];
    let treat_var = &args[2];
    let outcome_var = &args[3];

    // Execute matching algo
    let match_data = import_data(path).unwrap();
    let (att, att_variance, treat, control) = matches(
        &match_data,
        treat_var,
        outcome_var
    ).unwrap();
    let strike = Strike {
        att,
        att_variance,
        treat,
        control,
    };

    // Display the ATT and corresponding 95% confidence interval
    println!("{}", strike);
    ()
}
