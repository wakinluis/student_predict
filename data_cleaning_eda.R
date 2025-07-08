# Melo Villanueva – Data Ingestion & EDA
# Project: Predicting Students’ Dropout and Academic Success
# Description: Loads and cleans dataset, performs EDA, generates plots

# ─────────────────────────────
# 1. Load Libraries and Dataset
# ─────────────────────────────
library(tidyverse)

# Load dataset (semicolon-separated)
df <- read_delim("data.csv", delim = ";")

# ─────────────────────────────
# 2. Explore Target Distribution
# ─────────────────────────────

# View class counts
table(df$Target)

# Visualize student status distribution
status_plot <- ggplot(df, aes(x = Target)) +
  geom_bar(fill = "#1f77b4") +
  labs(
    title = "Distribution of Student Status",
    x = "Status",
    y = "Count"
  ) +
  theme_minimal()

# Save plot
ggsave("plots/EDA_Distribution_Student_Status.jpg", plot = status_plot, width = 8, height = 6, dpi = 300)

# ─────────────────────────────
# 3. Data Cleaning
# ─────────────────────────────

# Fix incorrect column name with hidden tab
colnames(df)[colnames(df) == "Daytime/evening attendance\t"] <- "Daytime/evening attendance"

# Convert numeric-like columns to categorical (factor)
df <- df %>%
  mutate(
    `Tuition fees up to date` = as.factor(`Tuition fees up to date`),
    `Scholarship holder` = as.factor(`Scholarship holder`),
    `Daytime/evening attendance` = as.factor(`Daytime/evening attendance`),
    `Previous qualification` = as.factor(`Previous qualification`),
    `Mother's qualification` = as.factor(`Mother's qualification`),
    `Father's qualification` = as.factor(`Father's qualification`),
    `Mother's occupation` = as.factor(`Mother's occupation`),
    `Father's occupation` = as.factor(`Father's occupation`),
    `Application mode` = as.factor(`Application mode`),
    `Application order` = as.factor(`Application order`)
  )

# ─────────────────────────────
# 4. Outlier Capping (1st–99th Percentile)
# ─────────────────────────────

# Define outlier capping function
cap_outliers <- function(x) {
  q1 <- quantile(x, 0.01, na.rm = TRUE)
  q99 <- quantile(x, 0.99, na.rm = TRUE)
  x[x < q1] <- q1
  x[x > q99] <- q99
  return(x)
}

# Apply to selected numeric columns
df <- df %>%
  mutate(
    `Admission grade` = cap_outliers(`Admission grade`),
    `Curricular units 1st sem (grade)` = cap_outliers(`Curricular units 1st sem (grade)`),
    `Curricular units 2nd sem (grade)` = cap_outliers(`Curricular units 2nd sem (grade)`),
    `Previous qualification (grade)` = cap_outliers(`Previous qualification (grade)`),
    `Age at enrollment` = cap_outliers(`Age at enrollment`)
  )

# ─────────────────────────────
# 5. Exploratory Data Analysis (Post-Cleaning)
# ─────────────────────────────

# 5.1 Histograms of numeric variables
hist_plot <- df %>%
  select(where(is.numeric)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30, fill = "#2ca02c", color = "black") +
  facet_wrap(~variable, scales = "free", ncol = 4) +
  labs(
    title = "Distribution of Numeric Variables (Histograms)",
    x = "Value",
    y = "Count"
  ) +
  theme_minimal()

ggsave("plots/EDA_Histograms_Numeric_Variables.jpg", plot = hist_plot, width = 14, height = 10, dpi = 300)

# 5.2 Boxplots for outlier detection
box_plot <- df %>%
  select(where(is.numeric)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = "", y = value)) +
  geom_boxplot(fill = "#ff7f0e") +
  facet_wrap(~variable, scales = "free", ncol = 4) +
  labs(
    title = "Boxplots for Outlier Detection (Numeric Variables)",
    x = "",
    y = "Value"
  ) +
  theme_minimal()

ggsave("plots/EDA_Boxplots_Outlier_Detection.jpg", plot = box_plot, width = 14, height = 10, dpi = 300)

# ─────────────────────────────
# 6. Final Check (Structure & Summary)
# ─────────────────────────────

glimpse(df)
summary(df)
write_csv(df, "cleaned_data.csv")
