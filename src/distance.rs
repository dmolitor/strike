use polars::prelude::{ChunkCompare, DataFrame, PolarsResult};
use std::iter::zip;

// Nearest Neighbor match
//
// Given a DataFrame with propensity scores, this function will
// find which observation is the nearest neighbor to the provided `pscore`
// in terms of propensity score difference. The `strike_id` argument is
// used to ensure that an observation is never matched to itself as its
// nearest neighbor. If ties occur, it returns the first observation
// from the matches. The return value is a 1-row DataFrame of the
// nearest neighbor observation.
fn find_nn(data: &DataFrame, pscore: f64, strike_id: i64) -> PolarsResult<DataFrame> {
    //println!("strike_id: {}", &strike_id);
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

// Nearest Neighbor propensity score matching
//
// Given a `main` DataFrame and a `target` DataFrame, this function will find
// the nearest neighbor propensity score match from `target` for every row in
// `main`. The return value is a matched DataFrame where every row is the
// match for the corresponding row in `main`. E.g. the return DataFrame has
// the same # of rows as `main` and the first row is the matched observation
// for the first row in `main`.
pub fn nn_match(main: &DataFrame, target: &mut DataFrame) -> PolarsResult<DataFrame> {
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