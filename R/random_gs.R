library(broom)
library(lfe)
library(tidyverse)
library(data.table)

# Download data -----------------
us <- fread(file.path(DATA_PROCESSED, "df_us.csv")) |> mutate(country = "US")
global <- fread(file.path(DATA_PROCESSED, "df_gexus.csv")) |> mutate(country = "Global ex-US")
(cn_diff <- setdiff(colnames(us), colnames(global)))
# Add missing columns
global[, (cn_diff) := NA]
data <- rbind(us, global)
if (TRUE) {
  data <- rbind(us, global)
  # Add separate global sample
  data <- rbind(
    data,
    data |> mutate(country = "global")
  )
}
data[, country_date := paste0(excntry, "_", date)]
# ESG columns
esg_cols <- c("log_scope1tot", "log_scope12tot", "log_scope123tot", 
              "scope1int_sales", "scope12int_sales", "scope123int_sales", 
              "scope1int_assets", "scope12int_assets", "scope123int_assets", 
              "industry_esg_score", "weighted_esg_score", "e_score", 
              "greenness", "e_climate_score", "e_natres_score", 
              "e_waste_score", "e_envopps_score", "total_esg_score", 
              "environment_score", "release_total_sales", 
              "prod_waste_total_sales", "release_total_assets", 
              "prod_waste_total_assets")
# Exclude non-included columns
if (FALSE) {
  esg_cols <- c("scope1int_sales", "scope12int_sales", "scope123int_sales", 
                "scope1int_assets", "scope12int_assets", "scope123int_assets",
                #"industry_esg_score", 
                "weighted_esg_score", "E_score", 
                # "E_climate_score", ,
                #"greenness", "E_natres_score", 
                # "E_waste_score", "E_envopps_score", 
                # "release_total_sales", 
                # "prod_waste_total_sales", "release_total_assets", 
                # "prod_waste_total_assets",
                "total_esg_score", "environment_score"
              )
}
n_esg <- length(esg_cols)

# Create distribution of random averages ---------------------------------------
# Reproducibility: seed before the 1000 random-weight draws below. Without
# this, every run produces a different `random_score_regs` and therefore
# different downstream PDFs (random_icc_t.pdf, random_average_MT.pdf,
# random_average_distribution.pdf, icc_MT.pdf). Per-paper repos cloned
# from this commit onward inherit the seed automatically.
if (TRUE) {
  esg_mat <- data[, esg_cols, with = FALSE] |> as.matrix()
  set.seed(1)
  random_score_regs <- 1:1000 |> map(function(s) {
    # Random weights
    ws <- runif(n=n_esg, min=0, max=1)
    # Weight scale (only sum over non-missing measures)
    raw_scale <- as.integer(!is.na(esg_mat)) |> matrix(nrow=nrow(esg_mat), ncol=n_esg)
    raw_scale <- raw_scale %*% ws
    # Sum over non-missing measures
    raw <- esg_mat |> sweep(2, ws, "*")
    raw <- raw |> rowSums(na.rm=T)
    # Standardize sum of weights to zero
    raw <- raw/raw_scale
    # Standardize each month
    data$score_rnd <- raw
    data[, score_rnd := (score_rnd-mean(score_rnd, na.rm=T))/sd(score_rnd, na.rm=T), by = .(country, date)]
    
    data |> 
      filter(!is.na(score_rnd)) |>
      group_by(country) |>
      filter(country=="global") |> 
      # filter(score=="E_score") |> 
      nest() |> 
      mutate(
        fit = data |> 
          map(~felm(icc_avg ~ score_rnd + beta_252d + ebit_at + netdebt_at + log_be | country_date | 0 | date + gics, data=.x)) |> 
          map(broom::tidy) 
      ) |> 
      select(-data) |> 
      unnest(fit) |> 
      mutate(sim = s)
  }, .progress = T) |> 
    bind_rows()
  # Save
  # random_score_regs |> fwrite("Data/Generated/random_score_regs.csv")
  random_score_regs |> fwrite(file.path(DATA_INTERIM, "random_score_regs_new.csv")) # "_new" to avoid overwriting the old file used in the paper
} else {
  random_score_regs <- fread(file.path(DATA_INTERIM, "random_score_regs_new.csv"))
}

# Check the ew average
if (FALSE) {
  data[, esg_test := esg_mat |> rowMeans(na.rm = T)]
  data[, esg_test := (esg_test-mean(esg_test))/sd(esg_test), by = .(country, date)]
  felm(icc_avg ~ esg_test + beta_252d + ebit_at + netdebt_at + log_be | country_date | 0 | date + gics, data = data[country=="global"]) |> 
    broom::tidy() 
  data[, esg_test := NULL]
}
# Clean
data[, score_rnd := NULL]

# Random estimate distribution -------------------------------------------------
# felm(icc_avg ~ score_rnd   + beta_252d + ebit_at + netdebt_at + log_be | country_date | 0 | date + gics, data = data[country=="global"]) |> broom::tidy() |> filter(term == "score_rnd")
# felm(icc_avg ~ green_score + beta_252d + ebit_at + netdebt_at + log_be | country_date | 0 | date + gics, data = data[country=="global"]) |> broom::tidy() |> filter(term == "green_score")
base_est_reg <- felm(icc_avg ~ green_score + beta_252d + ebit_at + netdebt_at + log_be | country_date | 0 | date + gics, data = data[country=="global"])
base_est <- coef(base_est_reg)["green_score"]
base_est_t <- summary(base_est_reg)$coefficients["green_score", "t value"]
base_est_p <- summary(base_est_reg)$coefficients["green_score", "Pr(>|t|)"]

random_score_regs |>
  filter(term == "score_rnd") |>
  ggplot(aes(x = statistic)) +
  geom_histogram(bins = 25, fill = colours_theme[1]) +
  geom_vline(xintercept = base_est_t, linetype = "dotted") +
  # Horizontal arrow from just-left-of-the-label to just-right-of-the-vline.
  # Previous version had the segment only 0.10 t-stat units long, so the
  # arrowhead consumed nearly all of it and the line wasn't visible; this
  # version makes the connecting line clearly readable.
  annotate(
    "segment",
    x = base_est_t+0.38, xend = base_est_t+0.03,
    y = 95, yend = 95,
    arrow = arrow(length = unit(0.2, "cm")
    )) +
  annotate(
    "text",
    x = base_est_t+0.40,
    y = 95,
    label = "Our greenium\n t-statistic",
    hjust = 0,
    size = 2.75
  ) +
  labs(x = "t-statistic", y = "Count")
if (TRUE) {
  ggsave(file.path(OUTPUT_FIGURES, "random_icc_t.pdf"), width = 4, height = 2.5, device = "pdf")
}


random_score_regs |> 
  filter(term == "score_rnd") |> 
  ggplot(aes(x = estimate*100)) +
  geom_histogram(bins = 25, fill = colours_theme[1]) +
  geom_vline(xintercept = base_est*100, linetype = "dashed") +
  labs(x = "Greenium estimate", y = "Count")
if (TRUE) {
  ggsave(file.path(OUTPUT_FIGURES, "random_average_distribution.pdf"), width = 8, height = 4, device = "pdf")
}


random_score_regs |> 
  filter(term=="score_rnd") |>
  summarise(
    our_estimate = base_est,
    mean = mean(estimate),
    min = min(estimate),
    p25 = quantile(estimate, 0.25),
    p50 = quantile(estimate, 0.5),
    p75 = quantile(estimate, 0.75),
    max = max(estimate),
  )


# Multiple testing figure ------------------------------------------------------
random_score_regs |> 
  filter(term=="score_rnd") |> 
  # mutate(p.value = runif(n = 100, min = 0, max = 0.1)) |> 
  group_by(country) |> 
  mutate(
    p_rank = frank(p.value)/n(),
    p_bh = p.adjust(p.value, method = "BH"),
    sig_bh = if_else(p_bh<0.05, "significant", "insignificant")
  ) |> 
  ggplot(aes(p_rank, p.value, colour = sig_bh)) +
  geom_point() +
  geom_abline(slope=0.05, intercept=0, linetype="dashed") +
  labs(colour = "Multiple testing inference:") + 
  facet_wrap(~country) +
  theme(
    legend.position = "top"
  )

p_perc_data <- random_score_regs |>
  filter(term == "score_rnd") |> 
  group_by(country) |> 
  arrange(p.value) |> 
  mutate(
    p_rank_perc = frank(p.value) / n(),
    bh_threshold = p_rank_perc * 0.05,
    max_sig = max(p_rank_perc[p.value < bh_threshold]),
    sig_bh = if_else(
      p_rank_perc <= max_sig, 
      "significant", 
      "insignificant"
    ) |> factor(levels = c("significant", "insignificant"))
  )

base_est_p_perc <- ecdf(p_perc_data$p.value)(base_est_p)

if (FALSE) {
  p_perc_data |> 
    ggplot(aes(p_rank_perc, p.value, colour = sig_bh)) +
    geom_point() +
    coord_cartesian(ylim = c(0, 0.05)) +
    geom_abline(slope = 0.05, intercept = 0, linetype = "dashed") +
    # Add the first arrow pointing down (Benjamini-Hochberg threshold)
    annotate(
      "segment",
      x = 0.5, xend = 0.5,
      y = 0.035, yend = 0.026,
      arrow = arrow(length = unit(0.2, "cm"))
    ) +
    # Add the first text label
    annotate(
      "text",
      x = 0.5,
      y = 0.042,
      label = "Benjamini-Hochberg\nmultiple testing threshold",
      hjust = 0.5,
      size = 2.5
    ) +
    # Add the second arrow pointing down (Our greenium estimate)
    annotate(
      "segment",
      x = base_est_p_perc, xend = base_est_p_perc,
      y = base_est_p + 0.014, yend = base_est_p+0.002,
      arrow = arrow(length = unit(0.2, "cm")
      )) +
    # Add the second text label
    annotate(
      "text",
      x = base_est_p_perc,
      y = base_est_p + 0.018,
      label = "Our greenium\np-value",
      hjust = 0.25,
      size = 2.5
    ) +
    labs(y = "p-value", x = "p-value percentile") + 
    theme(
      legend.position = "none"
    )
} else {
  p_perc_data |> 
    ggplot(aes(p_rank_perc, p.value, colour = sig_bh)) +
    geom_point() +
    coord_cartesian(ylim = c(0, 0.05)) +
    geom_abline(slope = 0.05, intercept = 0, linetype = "dashed") +
    # Add the first arrow pointing down (Benjamini-Hochberg threshold)
    geom_vline(xintercept = base_est_p_perc, linetype = "dotted") +
    annotate(
      "segment",
      x = 0.5, xend = 0.5,
      y = 0.035, yend = 0.026,
      arrow = arrow(length = unit(0.2, "cm"))
    ) +
    # Add the first text label
    annotate(
      "text",
      x = 0.5,
      y = 0.04,
      label = "Benjamini-Hochberg\n multiple testing threshold",
      hjust = 0.5,
      size = 2.75
    ) +
    # Add the second arrow pointing down (Our greenium estimate)
    annotate(
      "segment",
      x = base_est_p_perc+0.06, xend = base_est_p_perc+0.02,
      y = 0.045, yend = 0.045,
      arrow = arrow(length = unit(0.2, "cm")
      )) +
    # Add the second text label
    annotate(
      "text",
      x = base_est_p_perc+0.18,
      y = 0.045,
      label = "Our greenium\np-value",
      hjust = 0.5,
      vjust = 0.5,
      size = 2.75
    ) +
    labs(y = "p-value", x = "p-value percentile") + 
    theme(
      legend.position = "none"
    )
}

if (TRUE) {
  ggsave(file.path(OUTPUT_FIGURES, "random_average_MT.pdf"), width = 4, height = 2.5, device = "pdf")
}

# highest t/p value
random_score_regs |> 
  filter(term=="score_rnd") |> 
  arrange(-statistic) |> 
  head(1)

# Multiple testing with the individual measures --------------------------------
separate_regs <- data[, c("id", "date", "country_date", "gics", "country",
                     "icc_avg", 
                     "beta_252d", "ebit_at", "netdebt_at", "log_be",
                     "green_score", esg_cols), with = F] |> 
  melt(measure.vars = c(esg_cols, "green_score"), variable.name = "score") |> 
  filter(!is.na(value)) |> 
  group_by(country, score) |>
  # filter(score=="E_score") |> 
  nest() |> 
  mutate(
    fit = data |> 
      map(~felm(icc_avg ~ value + beta_252d + ebit_at + netdebt_at + log_be | country_date | 0 | date + gics, data=.x)) |> 
      map(broom::tidy) 
  ) |> 
  unnest(fit)

separate_regs |> 
  filter(term=="value") |> 
  mutate(country = if_else(country=="global", "Global", country)) |>
  mutate(
    p_bh = round(p.adjust(p.value, method = "BH"), 5)
  ) |> 
  # ungroup() |>
  group_by(country) |>
  # mutate(
  #   sig_bh = if_else(p_bh < 0.05, "significant", "insignificant") |> factor(levels = c("significant", "insignificant")),
  #   p_rank = frank(p.value)/n(),
  #   type = if_else(score=="green_score", "main_score", "other")
  # ) |>
  arrange(p.value) |> 
  mutate(
    p_rank_perc = frank(p.value)/n(),
    bh_threshold = p_rank_perc * 0.05,
    max_sig = max(p_rank_perc[p.value<bh_threshold]),
    sig_bh = if_else(p_rank_perc <= max_sig, "significant", "insignificant") |> factor(levels = c("significant", "insignificant")),
    type = if_else(score=="green_score", "main_score", "other")
  ) |> 
  # ggplot(aes(x = reorder(score, p.value), y = p.value, color = sig_bh)) +
  ggplot(aes(x = p_rank_perc, y = p.value, color = sig_bh, shape = type)) +
  geom_point(size=2) + 
  geom_abline(slope=0.05, intercept=0, linetype="dashed") +
  facet_wrap(~country) +
  labs(colour = "Multiple testing inference:", x = "Percentile ranked p.values") + 
  # facet_wrap(~country) + 
  theme(
    legend.position = "top"
  ) +
  guides(shape = "none")


# New version ------------------------------------------------------------------
test <- separate_regs |> 
  filter(term == "value") |> 
  mutate(country = if_else(country == "global", "Global", country)) |>
  mutate(
    p_bh = round(p.adjust(p.value, method = "BH"), 5)
  ) |> 
  ungroup() |>
  arrange(p.value) |> 
  mutate(
    p_rank_perc = frank(p.value) / n(),
    bh_threshold = p_rank_perc * 0.05,
    max_sig = max(p_rank_perc[p.value < bh_threshold]),
    sig_bh = if_else(p_rank_perc <= max_sig, "Significant", "Insignificant") |> 
      factor(levels = c("Significant", "Insignificant")),
    type = if_else(score == "green_score", "main_score", "other")
  )

# Extract specific data points for the arrows
gs_us <- test |> filter(country == "US" & score == "green_score")
gs_global_ex <- test |> filter(country == "Global ex-US" & score == "green_score")
gs_global <- test |> filter(country == "Global" & score == "green_score")

# Prepare a data frame for the arrows and labels
arrows_df <- data.frame(
  x = c(gs_us$p_rank_perc, gs_global_ex$p_rank_perc, gs_global$p_rank_perc),
  y = c(gs_us$p.value, gs_global_ex$p.value, gs_global$p.value),
  label = c("Agg. green score\n(US)",
            "Agg. green score\n(Global ex-US)",
            "Agg. green score\n(Global)"),
  country = c("US", "Global ex-US", "Global"),
  sig_bh = rep("Significant", 3)
)

# Calculate positions for the arrows and labels
# y_max <- max(test$p.value)
arrows_df <- arrows_df %>%
  mutate(
    y_start = 0.2,               # Starting point of the arrow
    y_start = if_else(country == "Global", 0.4, y_start),  # Adjust the starting point for the Global label to avoid overlapo
    label_y = y_start + 0.05,      # Position of the text label
    label_x = if_else(country == "Global ex-US", x - 0.03, x),  # Adjust the x position for the Global ex label to avoid overlap
  )

# Create the plot with arrows and labels
test |> 
  ggplot(aes(x = p_rank_perc, y = p.value, color = sig_bh, shape = country)) +
  geom_point(size = 2) + 
  geom_abline(slope = 0.05, intercept = 0, linetype = "dashed") +
  labs(
    colour = "Multiple testing inference",
    x = "Percentile rank of p-value",
    y = "p-value",
    shape = "Region"
  ) + 
  theme(legend.position = "top") +
  geom_segment(
    data = arrows_df,
    aes(x = x, xend = x, y = y_start, yend = y+0.02),
    arrow = arrow(length = unit(0.2, "cm")),
    color = "black"
  ) +
  geom_text(
    data = arrows_df,
    aes(x = label_x, y = label_y, label = label),
    hjust = 0.5,
    size = 3.5,
    color = "black"
  ) 
ggsave(file.path(OUTPUT_FIGURES, "icc_MT.pdf"), width = 8, height = 4, device = "pdf")
