# Packages -----------------------------------------------------------------------------
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
    data |> mutate(country = "Global")
  )
}
data[, country_date := paste0(excntry, "_", date)]

# Crete standard errors with different clustering ---
# Issue: I don't have excntry
regs <- c("US", "Global") |> map(function(x) {
  sub <- data[country==x]
  c("0", "date", "id", "gics", "date+id", "date+gics")|> map(function(cl) { #, "boot"
    if (cl=="boot") {
      dqrng::dqset.seed(2352342)
      set.seed(1)
      lm_fit <- lm(icc_avg ~ green_score + beta_252d + ebit_at + netdebt_at + log_be + factor(date), data = sub[!is.na(date) & !is.na(gics)])
      boot_lm <- boottest(
        lm_fit,
        clustid = c("gics"), 
        param = "green_score",
        B = 999
      )
      boot_lm |> tidy() |> mutate(
        clusters=cl,
        term = "green_score",
        std.error = NA
      )
    } else {
      form <- paste0("icc_avg~green_score+ beta_252d + ebit_at + netdebt_at + log_be | country_date | 0 | ", cl) |> as.formula()
      felm(form, data=sub) |> 
        broom::tidy() |> 
        mutate(
          clusters=cl,
          conf.low = estimate - 1.96*std.error,
          conf.high = estimate + 1.96*std.error
        ) |> 
        select(term, estimate, statistic, p.value, std.error, conf.low, conf.high, clusters)
    }
  }, .progress = T) |> bind_rows() |> mutate(country=x)
}) |> bind_rows()

cl_name_fun <- function(x) {
  case_when(
    x=="0" ~ "None",
    x=="date" ~ "Time",
    x=="id" ~ "Firm",
    x=="gics" ~ "Industry",
    x=="boot" ~ "Industry (wild bootstrap)",
    x=="date+id" ~ "Time+firm",
    x=="date+gics" ~ "Time+industry",
  ) |> factor(levels = c("None", "Time", "Firm", "Industry", "Industry (wild bootstrap)", "Time+firm", "Time+industry"))
}

# Plot 
regs |> 
  filter(term == "green_score") |> 
  mutate(
    clusters = clusters |> cl_name_fun(),
    country = country |> factor(levels = c("Global", "US")),
    choice = if_else(clusters=="Time+industry", "Our choice", "Alternative choices") |> factor(levels = c("Our choice", "Alternative choices"))
  ) |> 
  ggplot(aes(clusters, estimate, fill = country)) +
  scale_y_reverse() +
  geom_col(position="dodge", alpha=0.95) +
  # geom_hline(aes(yintercept = estimate), linetype="dashed", color = "black") +
  geom_hline(aes(yintercept = 0), linetype="solid", color = "black") +
  geom_errorbar(aes(ymin=conf.low, ymax=conf.high), width=.3, position = position_dodge(width = 0.9)) +
  # facet_wrap(~country, scales="free_y", ncol = 1) +
  labs(x = "Clustering", y = "Equity greenium estimate with 95% CI", fill = "Region:") +
  theme(
    legend.position = c(0.05, 1),  # Move to top left corner
    legend.justification = c(0, 1),
    legend.title=element_blank(),
    legend.background = element_rect(color = "lightgrey", fill = "white", linewidth = 0.2)
  )
ggsave(file.path(OUTPUT_FIGURES, "clustering.pdf"), width = 8, height = 4, device = "pdf")

# OLD CODE 
if (FALSE) {
  # Standard errors 
  regs |> 
    filter(term == "green_score") |> 
    arrange(country, std.error) |> 
    group_by(country) |> 
    mutate(rel = std.error/std.error[clusters=="0"])

  # Wild bootstrap test ------------------------------------------------------------
  # set seed via dqset.seed for engine = "R" & Rademacher, Webb & Normal weights
  dqrng::dqset.seed(2352342)
  # set 'familiar' seed for all other algorithms and weight types 
  set.seed(1)

  lm_fit <- felm(icc_avg ~ green_score + beta_252d + ebit_at + netdebt_at + 
                  log_be | date | 0 | 0, data = data[!is.na(date) & !is.na(gics) & country=="US"])

  lm_fit <- lm(icc_avg ~ green_score + beta_252d + ebit_at + netdebt_at + log_be + factor(date), data = data[!is.na(date) & !is.na(gics) & country=="US"])
  boot_lm <- boottest(
    lm_fit,
    clustid = c("gics"), 
    param = "green_score",
    B = 999
  )
  summary(boot_lm)

  # Placebo regressions (by industry) ----------------------------------------------
  # A better approach ----
  region <- "US"
  industries <- unique(data[country==region, .(gics)])
  n_ind <- nrow(industries)
  sub <- data[region=="US", .(id, date, gics, icc_avg, green_score, beta_252d, ebit_at, netdebt_at)] |> na.omit()
  if (FALSE) {
    # Will take around 5 hours to run
    set.seed(1)
    plan(multisession, workers = 5)  # For 10 iterations, the time drop from 412 second with sequatial to 180 sec with 5 workers
    # plan(sequential)
    tictoc::tic()
    placebo_regs_ind <- 1:1000 |> future_map(
      .f = function(s) {
        new <- copy(industries)[, r := sign(rnorm(n_ind))]
        new <- new[sub, on = "gics"]
        new[, green_score := green_score*r]
        c("0", "date", "id", "gics", "date+id", "date+gics") |> map(function(cl) {
          form <- paste0("icc_avg ~ green_score | date | 0 | ", cl) |> as.formula()
          felm(form, data=new) |> broom::tidy() |> mutate(clusters=cl)
        }) |> bind_rows() |> mutate(s=s)
      }, 
      .progress = T, 
      .options = furrr_options(seed = TRUE)
    ) |> bind_rows()
    tictoc::toc()
    # Save
    placebo_regs_ind |> fwrite(paste0("Data/placebo_regs_ind_", region,".csv"))
  } else {
    placebo_regs_ind |> fread(paste0("Data/placebo_regs_ind_", region,".csv"))
  }

  # Overall rejection rates
  rr <- placebo_regs_ind |> 
    # mutate(clusters = clusters |> cl_name_fun()) |>
    group_by(clusters) |> 
    summarise(
      rr = mean(abs(statistic)>=1.96),
      avg_se = mean(std.error)
    ) |> setDT()

  rr |> mutate(clusters = clusters |> cl_name_fun()) 

  # Distribution 
  text_n <- paste0("None (", formatC(rr[clusters=="0"]$rr, format = "f", digits = 2), ")")
  text_t <- paste0("Time (", formatC(rr[clusters=="date"]$rr, format = "f", digits = 2), ")")
  text_f <- paste0("Firm (", formatC(rr[clusters=="gvkey"]$rr, format = "f", digits = 2), ")")
  text_i <- paste0("Industry (", formatC(rr[clusters=="gic_6"]$rr, format = "f", digits = 2), ")")
  text_t_f <- paste0("Time+firm (", formatC(rr[clusters=="date+gvkey"]$rr, format = "f", digits = 2), ")")
  text_t_i <- paste0("Time+industry (", formatC(rr[clusters=="date+gic_6"]$rr, format = "f", digits = 2), ")")
  placebo_regs_ind |> 
    # select(s, clusters, statistic) |> 
    # pivot_longer(-s) |>
    mutate(
      name = case_when(
        clusters == "0" ~ text_n,
        clusters == "date" ~ text_t,
        clusters == "gvkey" ~ text_f,
        clusters == "gic_6" ~ text_i,
        clusters == "date+gvkey" ~ text_t_f,
        clusters == "date+gic_6" ~ text_t_i
      ) |> factor(levels = c(text_n, text_t, text_f, text_i, text_t_f, text_t_i))
    ) |> 
    ggplot(aes(x = statistic)) +
    geom_histogram() +
    facet_wrap(~name, scales="free_x") 

}
