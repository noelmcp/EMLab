# Load required libraries
library(EMLab)       # Our custom EM implementation
library(mclust)      # Standard GMM implementation
library(MASS)        # For generating synthetic data
library(ggplot2)     # For visualization
library(gridExtra)   # For plotting multiple figures

# Set seed for reproducibility
set.seed(45600)

# Generate synthetic data: Two Gaussian clusters
X <- rbind(
  MASS::mvrnorm(100, c(3, 3), diag(2)),
  MASS::mvrnorm(100, c(-3, -3), diag(2))
)

# Convert to matrix format
X_mat <- as.matrix(X)

# Run our custom EM implementation
result_custom <- em_gmm(X_mat, k = 2)

# Extract means and weights from our implementation
means_custom <- result_custom$means
weights_custom <- result_custom$weights

# Run mclust's EM implementation
result_mclust <- Mclust(X, G = 2)

# Extract means and weights from mclust
means_mclust <- result_mclust$parameters$mean
weights_mclust <- result_mclust$parameters$pro

# Print comparison of means and weights
cat("\n### Comparison of Estimated Means ###\n")
print("Custom Implementation:")
print(means_custom)

print("mclust Implementation:")
print(means_mclust)

cat("\n### Comparison of Cluster Weights ###\n")
print("Custom Implementation:")
print(weights_custom)

print("mclust Implementation:")
print(weights_mclust)

# Assign cluster labels
labels_custom <- apply(result_custom$responsibilities, 1, which.max)
labels_mclust <- result_mclust$classification

# Convert to data frame for plotting
df <- data.frame(X, ClusterCustom = factor(labels_custom), ClusterMclust = factor(labels_mclust))

# Plot clustering results
p1 <- ggplot(df, aes(X1, X2, color = ClusterCustom)) + 
  geom_point(size = 3) + 
  ggtitle("Custom EM Algorithm Clustering") + 
  theme_minimal()

p2 <- ggplot(df, aes(X1, X2, color = ClusterMclust)) + 
  geom_point(size = 3) + 
  ggtitle("mclust Clustering") + 
  theme_minimal()

# Display both plots
grid.arrange(p1, p2, ncol = 2)
