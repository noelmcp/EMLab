#include <RcppArmadillo.h>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]

// Function to initialize cluster means randomly
arma::mat initialize_means(const arma::mat& X, int k) {
  int n = X.n_rows;
  arma::uvec indices = arma::randperm(n, k);
  return X.rows(indices);
}

// Multivariate Gaussian density function
arma::vec gaussian_pdf(const arma::mat& X, const arma::vec& mean, const arma::mat& cov) {
  int d = X.n_cols;
  double det_cov = arma::det(cov);
  
  if (det_cov <= 1e-6) {
    det_cov = 1e-6;  // Avoid numerical instability
  }
  
  arma::mat inv_cov = arma::inv(cov + 1e-6 * arma::eye(d, d)); // Regularization
  
  arma::vec densities(X.n_rows);
  for (size_t i = 0; i < X.n_rows; i++) {
    arma::rowvec diff = X.row(i) - mean.t();
    double exponent = -0.5 * arma::as_scalar(diff * inv_cov * diff.t());
    densities(i) = std::exp(exponent) / std::sqrt(std::pow(2 * M_PI, d) * det_cov);
  }
  return densities;
}

// [[Rcpp::export]]
Rcpp::List em_gmm(const arma::mat& X, int k, int max_iter = 100, double tol = 1e-6) {
  int n = X.n_rows, d = X.n_cols;
  
  // Initialize parameters
  arma::mat means = initialize_means(X, k);
  std::vector<arma::mat> covariances(k, arma::eye(d, d)); // Identity covariance
  arma::vec weights = arma::ones(k) / k;
  arma::mat responsibilities(n, k, arma::fill::zeros);
  
  double log_likelihood_old = -arma::datum::inf;
  
  for (int iter = 0; iter < max_iter; iter++) {
    // E-step: Compute responsibilities
    for (int j = 0; j < k; j++) {
      responsibilities.col(j) = weights(j) * gaussian_pdf(X, means.row(j).t(), covariances[j]);
    }
    
    // Normalize responsibilities (avoid division by zero)
    arma::vec row_sums = arma::sum(responsibilities, 1);
    row_sums.elem(arma::find(row_sums == 0)).fill(1e-6); // Avoid zeros
    responsibilities.each_col() /= row_sums;
    
    // Compute log-likelihood
    double log_likelihood = arma::sum(arma::log(row_sums));
    
    // Check for convergence
    if (std::abs(log_likelihood - log_likelihood_old) < tol) {
      break;
    }
    log_likelihood_old = log_likelihood;
    
    // M-step: Update parameters
    arma::vec Nk = arma::sum(responsibilities, 0).t();
    weights = Nk / n;
    
    for (int j = 0; j < k; j++) {
      arma::mat new_mean(1, d);  // Explicitly define as a row vector
      new_mean.row(0) = (responsibilities.col(j).t() * X) / Nk(j);
      
      arma::mat new_cov = arma::zeros(d, d);
      for (int i = 0; i < n; i++) {
        arma::rowvec diff = X.row(i) - new_mean;
        new_cov += responsibilities(i, j) * (diff.t() * diff);
      }
      new_cov /= Nk(j);
      new_cov += 1e-6 * arma::eye(d, d); // Regularization
      
      means.row(j) = new_mean;
      covariances[j] = new_cov;
    }

  }
  
  return Rcpp::List::create(
    Rcpp::Named("means") = means,
    Rcpp::Named("covariances") = covariances,
    Rcpp::Named("weights") = weights,
    Rcpp::Named("responsibilities") = responsibilities
  );
}
