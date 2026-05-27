library(tidyverse)
library(data.table)

# Load data --------------------------------------------------------------------
data <- rbind(
  fread(file.path(DATA_PROCESSED, "factor_globalexus_vwcap.csv")) |> mutate(country="global_ex_us", industry=F) |> select(-V1),
  fread(file.path(DATA_PROCESSED, "factor_globalexus_vwcap_indn.csv")) |> mutate(country="global_ex_us", industry=T) |> select(-V1),
  fread(file.path(DATA_PROCESSED, "factor_us_vwcap.csv")) |> mutate(country="us", industry=F) |> rename("esg"=V1),
  fread(file.path(DATA_PROCESSED, "factor_us_vwcap_indn.csv")) |> mutate(country="us", industry=T) |> rename("esg"=V1),
  fill=T, use.names = T
)

# To long
data <- data |> 
  select(-SR) |> 
  rename("a_none"=exc_ret, "t_a_none"=t_exc_ret, "p_a_none"=p_exc_ret) |>
  pivot_longer(cols = matches("^(a_|t_a_|p_a_)"), names_to = "stat") |>
  separate(stat, into = c("stat", "model"), sep = "a_") |>
  mutate(
    stat = if_else(stat=="", "a", stat),
    stat = stat |> str_remove("_")
  ) |> 
  pivot_wider(names_from = stat, values_from = value) |> 
  mutate(
    country = case_when(
      country=="us" ~ "US",
      country=="global_ex_us" ~ "Global ex-US"
    ) |> factor(levels = c("US", "Global ex-US"))
  )

# Re-compute the old figures -----------------------------------------------
data |> 
  ggplot(aes(x = t)) + 
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = colours_theme[1], colour="white") +
  stat_function(fun = dnorm, args = list(mean = 0, sd = 1), col = "black", size = 0.5) +
  facet_wrap(~country) 

# Create the new figure --------------------------------------------------------
# By country
data |> 
  filter(!is.na(p) & country=="US") |> 
  group_by(country) |> 
  mutate(
    p_rank_perc = frank(p) / n(),
    bh_threshold = p_rank_perc * 0.05,
    test = sum(p<bh_threshold),
    max_sig = if_else(sum(p<bh_threshold)==0, 0, max(p_rank_perc[p < bh_threshold])),
    sig_bh = if_else(
      p_rank_perc <= max_sig, 
      "Significant", 
      "Insignificant"
    ) |> factor(levels = c("Significant", "Insignificant"))
  ) |> 
  setDT() |> 
  rbind(
    data.table(country="US", p_rank_perc=NA, p=NA, sig_bh="Significant"), 
    fill=T, use.names=T
  ) |> 
  ggplot(aes(p_rank_perc, p, colour = sig_bh)) +
  geom_point(alpha = 0.9, size=1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
  coord_cartesian(ylim = c(0, 1)) +
  geom_abline(slope = 0.05, intercept = 0, linetype = "dashed") +
  # Add the first arrow pointing down (Benjamini-Hochberg threshold)
  annotate(
    "segment",
    x = 0.75, xend = 0.75,
    y = 0.17, yend = 0.05,
    arrow = arrow(length = unit(0.2, "cm"))
  ) +
  # Add the first text label
  annotate(
    "text",
    x = 0.75,
    y = 0.2,
    label = "Benjamini-Hochberg 5% threshold",
    hjust = 0.5,
    size = 2.5
  ) +
  labs(y = "p-value", x = "p-value percentile", colour = "Multiple testing inference") + 
  theme(
    legend.position = "none"
  )
ggsave(file.path(OUTPUT_FIGURES, "realized_MT_us.pdf"), width = 4, height = 2.5, device = "pdf")

data |> 
  filter(!is.na(p) & country=="Global ex-US") |> 
  group_by(country) |> 
  mutate(
    p_rank_perc = frank(p) / n(),
    bh_threshold = p_rank_perc * 0.05,
    test = sum(p<bh_threshold),
    max_sig = if_else(sum(p<bh_threshold)==0, 0, max(p_rank_perc[p < bh_threshold])),
    sig_bh = if_else(
      p_rank_perc <= max_sig, 
      "Significant", 
      "Insignificant"
    ) |> factor(levels = c("Significant", "Insignificant"))
  ) |> 
  setDT() |> 
  rbind(
    data.table(country="US", p_rank_perc=NA, p=NA, sig_bh="Significant"), 
    fill=T, use.names=T
  ) |> 
  ggplot(aes(p_rank_perc, p, colour = sig_bh)) +
  geom_point(alpha = 0.05, size=1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
  coord_cartesian(ylim = c(0, 1)) +
  geom_abline(slope = 0.05, intercept = 0, linetype = "dashed") +
  # Add the first arrow pointing down (Benjamini-Hochberg threshold)
  annotate(
    "segment",
    x = 0.75, xend = 0.75,
    y = 0.17, yend = 0.05,
    arrow = arrow(length = unit(0.2, "cm"))
  ) +
  # Add the first text label
  annotate(
    "text",
    x = 0.75,
    y = 0.2,
    label = "Benjamini-Hochberg 5% threshold",
    hjust = 0.5,
    size = 2.5
  ) +
  labs(y = "p-value", x = "p-value percentile", colour = "Multiple testing inference") + 
  theme(
    legend.position = "none"
  )
ggsave(file.path(OUTPUT_FIGURES, "realized_MT_global_ex_us.pdf"), width = 4, height = 2.5, device = "pdf")

data |> 
  filter(!is.na(p)) |> 
  mutate(
    p_rank_perc = frank(p) / n(),
    bh_threshold = p_rank_perc * 0.05,
    test = sum(p<bh_threshold),
    max_sig = if_else(sum(p<bh_threshold)==0, 0, max(p_rank_perc[p < bh_threshold])),
    sig_bh = if_else(
      p_rank_perc <= max_sig, 
      "Significant", 
      "Insignificant"
    ) |> factor(levels = c("Significant", "Insignificant"))
  ) |> 
  setDT() |> 
  rbind(
    data.table(country="US", p_rank_perc=NA, p=NA, sig_bh="Significant"), 
    fill=T, use.names=T
  ) |> 
  ggplot(aes(p_rank_perc, p, colour = sig_bh)) +
  geom_point() +
  scale_y_continuous(breaks = c(0, 0.05, seq(0.25, 1, 0.25))) + 
  geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
  coord_cartesian(ylim = c(0, 1)) +
  geom_abline(slope = 0.05, intercept = 0, linetype = "dashed") +
  # Add the first arrow pointing down (Benjamini-Hochberg threshold)
  annotate(
    "segment",
    x = 0.5, xend = 0.5,
    y = 0.07, yend = 0.026,
    arrow = arrow(length = unit(0.2, "cm"))
  ) +
  # Add the first text label
  annotate(
    "text",
    x = 0.5,
    y = 0.1,
    label = "Benjamini-Hochberg\nmultiple testing threshold",
    hjust = 0.5,
    size = 3.5
  ) +
  labs(y = "p-value", x = "p-value percentile", colour = "Multiple testing inference") + 
  theme(
    legend.position = "top"
  )
ggsave(file.path(OUTPUT_FIGURES, "realized_MT_global.pdf"), width = 8, height = 2.5, device = "pdf")

# t --------------------------------------------------------------------
data |> 
  filter(country == "US") |> 
  ggplot(aes(x = t)) +
  # Histogram for US with 20 bins
  geom_histogram(bins = 20, fill = colours_theme[1], colour="white") +
  labs(x = "t-statistic", y = "Count")
ggsave(file.path(OUTPUT_FIGURES, "realized_t_us.pdf"), width = 4, height = 2.5, device = "pdf")

data |> 
  filter(country == "Global ex-US") |> 
  ggplot(aes(x = t)) +
  # Histogram for US with 20 bins
  geom_histogram(bins = 50, fill = colours_theme[1], colour="white") +
  labs(x = "t-statistic", y = "Count")
ggsave(file.path(OUTPUT_FIGURES, "realized_t_global_ex_us.pdf"), width = 4, height = 2.5, device = "pdf")

data |> 
  ggplot(aes(x = t)) +
  # Histogram for US with 20 bins
  geom_histogram(bins = 50, fill = colours_theme[1], colour="white") +
  labs(x = "t-statistic", y = "Count")
ggsave(file.path(OUTPUT_FIGURES, "realized_t_global.pdf"), width = 4, height = 2.5, device = "pdf")
