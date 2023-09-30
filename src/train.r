# Required Libraries
library(jsonlite)
library(dplyr)
library(tidyr)
library(readr)
library(fastDummies)
library(lightgbm)

# Define directories and paths

ROOT_DIR <- dirname(getwd())
MODEL_INPUTS_OUTPUTS <- file.path(ROOT_DIR, 'model_inputs_outputs')
INPUT_DIR <- file.path(MODEL_INPUTS_OUTPUTS, "inputs")
INPUT_SCHEMA_DIR <- file.path(INPUT_DIR, "schema")
DATA_DIR <- file.path(INPUT_DIR, "data")
TRAIN_DIR <- file.path(DATA_DIR, "training")
MODEL_ARTIFACTS_PATH <- file.path(MODEL_INPUTS_OUTPUTS, "model", "artifacts")
OHE_ENCODER_FILE <- file.path(MODEL_ARTIFACTS_PATH, 'ohe.rds')
PREDICTOR_FILE_PATH <- file.path(MODEL_ARTIFACTS_PATH, "predictor", "predictor.rds")
IMPUTATION_FILE <- file.path(MODEL_ARTIFACTS_PATH, 'imputation.rds')
TOP_10_CATEGORIES_MAP <- file.path(MODEL_ARTIFACTS_PATH, "top_10_map.rds")
COLNAME_MAPPING <- file.path(MODEL_ARTIFACTS_PATH, "colname_mapping.csv")
SCALING_FILE <- file.path(MODEL_ARTIFACTS_PATH, "scaler.rds")
LABEL_ENCODER_FILE <- file.path(MODEL_ARTIFACTS_PATH, 'label_encoder.rds')
ENCODED_TARGET_FILE <- file.path(MODEL_ARTIFACTS_PATH, "encoded_target.rds")
REMOVED_COLUMNS_FILE <- file.path(MODEL_ARTIFACTS_PATH, "removed_column_names_list.txt")



if (!dir.exists(MODEL_ARTIFACTS_PATH)) {
    dir.create(MODEL_ARTIFACTS_PATH, recursive = TRUE)
}
if (!dir.exists(file.path(MODEL_ARTIFACTS_PATH, "predictor"))) {
    dir.create(file.path(MODEL_ARTIFACTS_PATH, "predictor"))
}


# Reading the schema
# The schema contains metadata about the datasets. 
# We will use the scehma to get information about the type of each feature (NUMERIC or CATEGORICAL)
# and the id and target features, this will be helpful in preprocessing stage.

file_name <- list.files(INPUT_SCHEMA_DIR, pattern = "*.json")[1]
schema <- fromJSON(file.path(INPUT_SCHEMA_DIR, file_name))
features <- schema$features

numeric_features <- features$name[features$dataType == "NUMERIC"]
categorical_features <- features$name[features$dataType == "CATEGORICAL"]
id_feature <- schema$id$name
target_feature <- schema$target$name
model_category <- schema$modelCategory
nullable_features <- features$name[features$nullable == TRUE]

# Reading training data
file_name <- list.files(TRAIN_DIR, pattern = "*.csv")[1]
# Read the first line to get column names
header_line <- readLines(file.path(TRAIN_DIR, file_name), n = 1)
col_names <- unlist(strsplit(header_line, split = ",")) # assuming ',' is the delimiter
# Read the CSV with the exact column names
df <- read.csv(file.path(TRAIN_DIR, file_name), skip = 0, col.names = col_names, check.names=FALSE)

# Data Preprocessing
# Impute missing data
imputation_values <- list()
for (column in nullable_features) {
    # Create missing indicator
    missing_indicator_col_name <- paste(column, "is_missing", sep="_")
    df[[missing_indicator_col_name]] <- ifelse(is.na(df[[column]]), 1, 0)
    
    if (column %in% numeric_features) {
        value <- median(df[, column], na.rm = TRUE)
    } else {
        value <- as.character(df[, column] %>% tidyr::replace_na())
        value <- value[1]
    }
    df[, column][is.na(df[, column])] <- value
    imputation_values[column] <- value
}
saveRDS(imputation_values, IMPUTATION_FILE)


# Encoding Categorical features

# The id column is just an identifier for the training example, so we will exclude it during the encoding phase.
# Target feature will be label encoded in the next step.

ids <- df[, id_feature]
target <- df[, target_feature]
df <- df %>% select(-all_of(c(id_feature, target_feature)))

# One Hot Encoding
if(length(categorical_features) > 0){
    top_10_map <- list()
    for(col in categorical_features) {
        # Get the top 10 categories for the column
        top_10_categories <- names(sort(table(df[[col]]), decreasing = TRUE)[1:10])

        # Save the top 3 categories for this column
        top_10_map[[col]] <- top_10_categories
        # Replace categories outside the top 3 with "Other"
        df[[col]][!(df[[col]] %in% top_10_categories)] <- "Other"
    }

    df_encoded <- dummy_cols(df, select_columns = categorical_features, remove_selected_columns = TRUE)
    encoded_columns <- setdiff(colnames(df_encoded), colnames(df))
    saveRDS(encoded_columns, OHE_ENCODER_FILE)
    saveRDS(top_10_map, TOP_10_CATEGORIES_MAP)
    df <- df_encoded
}

# Remove constant columns
constant_columns <- which(apply(df, 2, var) == 0)
removed_column_names <- colnames(df)[constant_columns]

if (length(removed_column_names) > 0) {
    df <- df[,-constant_columns]
}

# Update numeric_features to exclude the removed columns
numeric_features <- setdiff(numeric_features, removed_column_names)

# Save removed columns in a file to use in predict.r
writeLines(removed_column_names, con = REMOVED_COLUMNS_FILE)

# Standard Scaling
scaling_values <- list()
for (feature in numeric_features) {
    feature_mean <- mean(df[[feature]], na.rm = TRUE)
    feature_std <- sd(df[[feature]], na.rm = TRUE)
    scaling_values[[feature]] <- list(mean = feature_mean, std = feature_std)
    # Standardize the feature
    df[[feature]] <- (df[[feature]] - feature_mean) / feature_std
}
# Save the scaling values for use during testing
saveRDS(scaling_values, SCALING_FILE)

# Cap outliers
lower_bound <- -4
upper_bound <- 4
for (feature in numeric_features) {
    df[[feature]] <- ifelse(df[[feature]] < lower_bound, lower_bound, df[[feature]])
    df[[feature]] <- ifelse(df[[feature]] > upper_bound, upper_bound, df[[feature]])
}


# Sanitize column names with special characters or spaces
sanitize_colnames <- function(names_vector) {
  # Check for unique original column names
  if (any(duplicated(names_vector))) {
    stop("Error: Given column names are not unique!")
  }

  # Trim spaces from column names
  names_vector <- trimws(names_vector)

  # Special characters sanitization
  sanitized_names <- gsub(" ", "_", names_vector)
  sanitized_names <- gsub("[^[:alnum:]_]", "_", sanitized_names)

  # Prefix with "feat_" - this is to get around columns that start with numbers
  sanitized_names <- paste0("feat_", sanitized_names)
  
  # Ensure uniqueness
  while(any(duplicated(sanitized_names))) {
    dupes <- table(sanitized_names)
    dupes <- as.character(names(dupes[dupes > 1]))
    
    for(d in dupes) {
      indices <- which(sanitized_names == d)
      sanitized_names[indices] <- paste0(d, "_", seq_along(indices))
    }
  }  
  return(sanitized_names)
}

# save the column name mapping to a file
new_colnames <- sanitize_colnames(colnames(df))
colname_mapping <- data.frame(
  original = colnames(df),
  sanitized = new_colnames
)
write.csv(colname_mapping, COLNAME_MAPPING, row.names = FALSE)

# apply new column names to df
colnames(df) <- new_colnames

# Label encoding target feature
levels_target <- schema$target$classes
encoded_target <- as.integer(factor(target, levels = levels_target)) - 1
saveRDS(levels_target, LABEL_ENCODER_FILE)
saveRDS(encoded_target, ENCODED_TARGET_FILE)


# Prepare the data for LightGBM
df_matrix <- data.matrix(df)
train_data <- lgb.Dataset(data = df_matrix, label = encoded_target)

# Set the parameters for LightGBM
if (model_category == 'binary_classification'){
    # Parameters for binary classification
    params <- list(
        objective = "binary",
        metric = "binary_logloss",
        boost_from_average = TRUE,
        learning_rate = 0.3,
        num_leaves = 31,
        feature_fraction = 0.9,
        bagging_fraction = 0.8,
        bagging_freq = 5
    )
    

} else if (model_category == "multiclass_classification") {
    # Parameters for multiclass classification
    params <- list(
        objective = "multiclass",
        num_class = length(unique(encoded_target)),
        metric = "multi_logloss",
        boost_from_average = FALSE,
        learning_rate = 0.3,
        num_leaves = 31,
        feature_fraction = 0.9,
        bagging_fraction = 0.8,
        bagging_freq = 5
    )
}

num_rounds <- 250

# Train the LightGBM classifier
model <- lgb.train(params, train_data, num_rounds, verbose = -1)

# Save the model using lgb.save
lgb.save(model, PREDICTOR_FILE_PATH)
