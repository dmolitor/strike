# strike

Strike implements 1:1, nearest neighbor, propensity score matching with replacement.
The target estimand is the Average Treatment Effect on the Treated (ATT),
which is the average effect of treatment for those who receive treatment:
$$\frac{1}{N^T}\sum_{i = 1}^{N^T}(Y_i(1) - Y_i(0)).$$ Strike also implements the
consistent estimator for the variance of the matching estimator for the ATT, as
proposed in "Large Sample Properties of Matching Estimators for Average
Treatment Effects" (Abadie and Imbens, 2006).

## Example

We can use propensity score matching (PSM) to estimate the effect of smoking on
psychological health using observational data. In this setup, the treatment
variable is smoking; the outcome variable is psychological distress, measured
on a unit interval ranging from 10 to 50. PSM matches (with replacement) each
smoker in the dataset to a non-smoker based on their propensity to be a smoker.
Propensity scores are estimated using logistic regression with a number of
potentially counfounding covariates, including sex, indigenous status,
education level (high school completion), marital status,
region of residence (major cities, inner regional or outer regional),
language background (English speaking or not), age, and risky alcohol
use (Yes/ No).

Strike will take care of all the implementation details under the hood as long
as all the covariates/treatment/outcome are numeric. To estimate the ATT,
we simply need to execute the Strike binary and pass it the path to the
data file as well as the names of the treatment indicator and outcome variable.

```rust
cargo build
 
cargo run -- ./examples/smoking.csv smoker psyc_distress
// STRIKE =======================================
// 
// # Treat: 974 | # Control (distinct): 736
//
// ATT                     : 3.386
// Variance                : 0.079
// 95% Confidence Interval : (2.835, 3.937)
```

On average, the effect of smoking among smokers is an increase in psychological
distress by ~3.4 units, with a 95% confidence interval that excludes 0.
