library(arrow)
library(tidyverse) 
library(data.table)

# Get baseline green estimate ----------------------------------------------------------
cols <- c("id", "date", "excntry", "me", "gics", "icc_avg", "green_score", "beta_252d", 
          "ebit_at", "netdebt_at", "log_be", "ret_exc_lead1m")
reg_data <- read_parquet(file.path(DATA_PROCESSED, "df_full.parquet"),
                     col_select = all_of(cols)) |> setDT()
reg_data[, date := as.Date(date)]
reg_data[, country_date := paste0(excntry, "_", date)]
greenium_reg <- felm(icc_avg ~ green_score + beta_252d + ebit_at + netdebt_at + 
                       log_be | country_date | 0 | date + gics, data = reg_data)
greenium_est <- coef(greenium_reg)["green_score"]

# Green ratings ------------------
global <- T
if (global) {
  green_scores <- copy(reg_data)
} else {
  green_scores <- fread(file.path(DATA_PROCESSED, "df_us.csv"))
  green_scores[, V1 := NULL]
}
green_scores |> setnames("date", "eom")

# NYSE breakpoints ---------------
nyse_bp <- fread(file.path(DATA_RAW, "other", "ME_Breakpoints.csv"))
nyse_bp[, eom := paste0(date, "01") |> fast_strptime("%Y%m%d") |> as.Date() |> ceiling_date(unit = "month")-1]
nyse_bp <- nyse_bp[, .(eom, nyse_p80=nyse_80)]

green_scores <- nyse_bp[, .(eom, nyse_p80)][green_scores, on = "eom"]
green_scores[, me_cap := pmin(me, nyse_p80)]

# Gics naming --------------------
gics_names <- readxl::read_xlsx(file.path(DATA_RAW, "other", "gics_names.xlsx"), range = "A4:H344", sheet="Effective close of Mar 17 2023") |>
  select("gics"=`Sub-Industry`, "name"=...8) |> 
  filter(!is.na(gics)) |> 
  mutate(gics = gics |> as.double()) |> 
  setDT()

# Firm naming --------------------
firm_names <- fread(file.path(DATA_RAW, "other", "crsp_dsenames.csv"))
firm_names <- firm_names[, .(id=permno, namedt, nameendt, ticker, comnam)]

# Combine with green scores
green_scores[, merge_date := eom]
green_scores <- firm_names[green_scores, on = .(id, namedt<=merge_date, nameendt>=merge_date)][, c("namedt", "nameendt") := NULL]

# Screen ------------------------
# To be consistent with GMB sample
green_scores <- green_scores[!is.na(green_score) & !is.na(me) & !is.na(ret_exc_lead1m)]

green_brown_industry_split <- FALSE
if (green_brown_industry_split) {
  # Split industries into three green groups
  green_scores[, pct := ecdf(green_score)(green_score), by = .(gics)]
  green_scores[, grp := case_when(
    pct <= 0.33 ~ "brown",
    pct <= 0.67 ~ "neutral",
    TRUE ~ "green"
  )]
} else {
  green_scores[, grp := "all"]
}
industries <- green_scores[, .(
  n = .N, 
  n_stocks = uniqueN(id),
  start = min(eom),
  end = max(eom),
  green_score = mean(green_score)#,
  # green_res = mean(green_res),
  # icc_avg = mean(icc_avg)
), by = .(gics, grp)][order(-green_score)][end==max(end)]
industries <- gics_names[industries, on = "gics"]
industries

if (FALSE) {
  # If we want to use recent values
  industries <- green_scores[eom==max(eom), .(
    n = .N, 
    n_stocks = uniqueN(id),
    start = min(eom),
    end = max(eom),
    green_score = mean(green_score)#,
    # green_res = mean(green_res),
    # icc_avg = mean(icc_avg)
  ), by = gics][order(-green_score)][end==max(end)]
  industries <- gics_names[industries, on = "gics"]
  industries
}
if(green_brown_industry_split) {
  industries <- industries[gics %in% c(55105020, 20302010, 10102050), .(
    type = case_when( 
      gics==55105020 ~ paste0("Renewables (", grp, ")"),
      gics==10102050 ~ paste0("Coal (", grp, ")"),
      gics==20302010 ~ paste0("Airlines (", grp, ")")
    ),
    gics,
    grp,
    green_score = green_score#,
    # green_res = green_res,
    # icc_avg = icc_avg
  )]
  industries <- industries[gics %in% c(55105020, 10102050) & 
                          grp %in% c("green", "brown")]
  industries[, c("gics", "grp") := NULL]
} else {
  industries <- industries[gics %in% c(55105020, 10102050)]
  industries[, type := case_when( 
      gics==55105020 ~ paste0("Renewables"),
      gics==10102050 ~ paste0("Coal"),
      gics==20302010 ~ paste0("Airlines")
    )]
  industries <- industries[, .(type, green_score)]
}





# Green-minus-brown ----------
green_scores[, pf := ceiling(3*ecdf(green_score)(green_score)), by = eom]
# Green score
gmb_pf <- green_scores[!is.na(me) & !is.na(green_score), .(
  # ew = mean(green_score),
  # vw = sum(green_score*me)/sum(me),
  green_score = sum(green_score*me_cap)/sum(me_cap)#,
  # green_res = sum(green_res*me_cap)/sum(me_cap),
  # icc_avg = sum(icc_avg*me_cap)/sum(me_cap)
), by = .(pf, eom)][, lapply(.SD, mean), by = pf][order(-pf)][, eom := NULL]
gmb_pf[, type := case_when(
  pf==1 ~ "Brown",
  pf==2 ~ "Neutral",
  pf==3 ~ "Green"
)]
gmb <- gmb_pf[, .(
  type = "Green-minus-Brown",
  green_score = green_score[pf==3]-green_score[pf==1]#,
  # green_res = green_res[pf==3]-green_res[pf==1],
  # icc_avg = icc_avg[pf==3]-icc_avg[pf==1]
)]
# Years to significant t-stat for US GMB
gmb_ret <- green_scores[!is.na(me) & !is.na(green_score) & excntry == "USA", .(
    ret = sum(ret_exc_lead1m*me_cap)/sum(me_cap),
    green_score = sum(green_score*me_cap)/sum(me_cap)
  ), by = .(pf, eom)][, .(
    ret = ret[pf==3]-ret[pf==1],
    green_score = green_score[pf==3]-green_score[pf==1]
  ), by = eom][, .(
    start = min(eom),
    end = max(eom),
    t = .N,
    ret = mean(ret)*12,
    sd = sd(ret)*sqrt(12),
    green_score = mean(green_score)
  )][, sr := ret/sd] #[, lapply(SD, function(x) formatC(x, digits=2, format="f"))]
print('US GMB annual return stats (%):')
print(gmb_ret)

# Expected return
exp_ret <- gmb_ret$green_score * greenium_est
print(paste0("Expected return for US GMB: ", round(exp_ret, 2), "%"))
# Expected SR
exp_sr <- exp_ret/gmb_ret$sd
print(paste0("Expected SR for US GMB: ", round(exp_sr, 2)))
# Expected t-stat
exp_t <- exp_sr * sqrt(gmb_ret$t/12)
print(paste0("Expected t-stat for US GMB: ", round(exp_t, 2)))
# Years to significant t-stat
yr_to_sig <- (1.96 / (exp_sr))^2
print(paste0("Years to 1.96 t-stat for US GMB: ", round(yr_to_sig, 2)))

# S&P500 ---------------------
if (FALSE) {
  green_scores[sp500==T, .N, by = eom] |> ggplot(aes(eom, N)) + geom_point()
  sp500 <- green_scores[sp500==T, .(
    green_score = sum(green_score*me)/sum(me)#,
    # green_res = sum(green_res*me)/sum(me),
    # icc_avg = sum(icc_avg*me)/sum(me)
  ), by = eom][, .(
    type="S&P500",
    green_score = mean(green_score)#,
    # green_res = mean(green_res),
    # icc_avg = mean(icc_avg)
  )]
}

market <- green_scores[!is.na(me), .(
  green_score = sum(green_score*me)/sum(me)#,
  # green_res = sum(green_res*me)/sum(me),
  # icc_avg = sum(icc_avg*me)/sum(me)
), by = eom][, .(
  type="Market (vw)",
  green_score = mean(green_score)#,
  # green_res = mean(green_res),
  # icc_avg = mean(icc_avg)
)]

# Average stock (just a sanity check)
avg_stock <- green_scores[, .(
  type="Market (ew)", 
  green_score=mean(green_score)
)]


# green_scores[eom==max(eom) & sp500==1][order(green_score)][,.(comnam, id, eom, me, green_score)] # Surprisingly, ratings agencies and FOX are at the top

apple <- green_scores[id==14593, .(type="Apple", green_score=mean(green_score))]
tesla <- green_scores[id==93436, .(type="Tesla", green_score=mean(green_score))]
google <- green_scores[id==90319, .(type="Google", green_score=mean(green_score))]
fb <- green_scores[id==13407, .(type="Facebook", green_score=mean(green_score))]
chevron <- green_scores[id==14541, .(type="Chevron", green_score=mean(green_score))]
exxon <- green_scores[id==11850, .(type="Exxon", green_score=mean(green_score))]

# ESG-tilted mutual funds ------------
if (TRUE) {
  # New approach based on sustainable funds from Morningstar
  esg_mf <- read_parquet(file.path(DATA_INTERIM, "holdings_agg.parquet")) |> setDT()
  esg_mf <- esg_mf[excntry == 'USA']
  esg_mf[, eom := date |> as.Date() |> ceiling_date(unit = "month")-1]
  esg_mf <- esg_mf[, .(id, eom, holding)]
  esg_mf <- green_scores[,.(id, eom, green_score)][esg_mf, on = .(id, eom)]
  esg_mf <- esg_mf[!is.na(green_score), .(
    n = .N,
    green_score = sum(green_score*holding)/sum(holding)
  ), by = .(eom)][n>=100, .(
    type = "Sustainable funds", 
    green_score = mean(green_score)
  )]
} else {
  # Old approach based on ESG classification from van der beck
  mf <- fread("Data/Green_Portfolio.csv") 
  mf[, V1 := NULL]
  mf |> setnames(old = "permno", new = "id")
  mf[, eom := as.Date(date)+months(1)-1]
  mf |> setnames(old = c("ESG Portfolio", "Mutual Fund Portfolio", "Revealed Preference ESG Score"), new = c("mf_esg", "mf_normal", "esg_tilt"))
  
  mf <- green_scores[,.(id, eom, green_score)][mf, on = .(id, eom)]
  # ESG mutual funds
  esg_mf <- mf[!is.na(green_score), .(
    n = .N,
    cov = sum(mf_esg),
    green_score = sum(green_score*mf_esg)/sum(mf_esg)#,
    # green_res = sum(green_res*mf_esg)/sum(mf_esg),
    # icc_avg = sum(icc_avg*mf_esg)/sum(mf_esg)
  ), by = .(eom)][, .(
    type = "ESG mutual funds", 
    green_score = mean(green_score)#,
    # green_res = mean(green_res),
    # icc_avg = mean(icc_avg)
  )]
  # Standard mutual funds
  all_mf <- mf[!is.na(green_score), .(
    n = .N,
    cov = sum(mf_normal),
    green_score = sum(green_score*mf_normal)/sum(mf_normal)#,
    # green_res = sum(green_res*mf_normal)/sum(mf_normal),
    # icc_avg = sum(icc_avg*mf_normal)/sum(mf_normal)
  ), by = .(eom)][, .(
    type = "All mutual funds", 
    green_score = mean(green_score)#,
    # green_res = mean(green_res),
    # icc_avg = mean(icc_avg)
  )]
}


# Figure -----------------------------------
data <- rbind(
  # sp500,
  market,
  avg_stock,
  industries,
  # apple,
  # chevron,
  # all_mf
  # esg_mf,
  gmb_pf[type=="Green"][, pf := NULL],
  gmb_pf[type=="Brown"][, pf := NULL]#,
  # gmbs
) |> 
  mutate(
    greenium = green_score * greenium_est
  )

# Greenium effects 
round(data[type=="Market (vw)"]$greenium, 2)
round(data[type=="Sustainable funds"]$greenium-data[type=="Market (vw)"]$greenium, 2)
round(data[type=="Green"]$greenium, 2)
round(data[type=="Brown"]$greenium, 2)
round(data[type=="Green"]$green_score-data[type=="Brown"]$green_score, 2)
round(data[type=="Green"]$greenium-data[type=="Brown"]$greenium, 2)
if (green_brown_industry_split) {
  round(data[type=="Renewables (green)"]$greenium, 2)
  round(data[type=="Coal (brown)"]$greenium, 2)
} else {
  round(data[type=="Renewables"]$greenium, 2)
  round(data[type=="Coal"]$greenium, 2)
}



# Figure for slides --------------
green_tilt_plot_fun <- function(data, types, offset_y=0.005, offset_y_line = 0.001, x_extend = 0.5) {
  # Add text and lines to plot manually
  sub <- data |> 
    mutate(greenium = greenium/100) |> 
    mutate(
      text_y = NA_real_,
      line_start = NA_real_,
      line_end = NA_real_
    ) |> 
    arrange(green_score)
  for (i in 1:nrow(data)) {
    if (i %% 2 == 1) {
      sub$text_y[i] <- sub$greenium[i] + offset_y
      sub$line_start[i] <- sub$greenium[i] + offset_y_line
      sub$line_end[i] <- sub$text_y[i] - offset_y_line
    } else {
      sub$text_y[i] <- sub$greenium[i] - offset_y
      sub$line_start[i] <- sub$greenium[i] - offset_y_line
      sub$line_end[i] <- sub$text_y[i] + offset_y_line
    }
    
  }
  x_min <- min(sub$green_score) - x_extend
  x_max <- max(sub$green_score) + x_extend
  y_min <- min(sub$text_y)
  y_max <- max(sub$text_y)
  
  if (types[1] == "none") {
    sub <- sub |> filter(type=="Market (ew)") |> mutate(across(.cols=c(green_score, greenium, text_y, line_start, line_end), .f = ~ NA_real_))
  } else {
    sub <- sub |> filter(type %in% types)
  }
  
  sub |> 
    ggplot(aes(green_score, (greenium), label=type, colour=green_score)) +
    scale_color_gradient2(low = "#654321", mid = "#808000", high = "#006400", midpoint = 0) + 
    scale_y_continuous(labels = scales::percent) +
    scale_x_continuous(breaks = seq(-5, 5, 1)) +
    geom_abline(intercept = 0, slope = greenium_est/100, linetype = "dashed", alpha=0.8, colour="grey") +
    geom_point(size=2) +
    coord_cartesian(xlim = c(x_min, x_max), ylim = c(y_min, y_max)) +
    annotate(geom="segment",
             x = sub$green_score,
             xend = sub$green_score,
             y = sub$line_start,
             yend = sub$line_end, colour = "grey") +
    geom_text(aes(y = text_y)) +
    labs(x = "Aggregate green score", y = "Greenium-induced annual expected return") +
    theme(
      legend.position = "none"
    )
}

# Market
types0 <- c("Market (ew)", "Market (vw)")
data |> green_tilt_plot_fun(types=types0)
ggsave(file.path(OUTPUT_FIGURES, "green_tilts0.pdf"), width = 8, height = 4, device = "pdf")
# ESG-tilted investors
types1 <- c(types0, "Sustainable funds")
data |> green_tilt_plot_fun(types=types1)
ggsave(file.path(OUTPUT_FIGURES, "green_tilts1.pdf"), width = 8, height = 4, device = "pdf")
# Aggresive E-tilted investors
types2 <- c(types1, "Green", "Brown")
data |> green_tilt_plot_fun(types=types2)
ggsave(file.path(OUTPUT_FIGURES, "green_tilts2.pdf"), width = 8, height = 4, device = "pdf")
# Firms 
if(green_brown_industry_split) {
  types3 <- c(types2, "Renewables (green)", "Renewables (brown)",
            "Coal (green)", "Coal (brown)")
} else {
  types3 <- c(types2, "Renewables", "Coal", "Airlines")
}
data |> green_tilt_plot_fun(types=types3)
ggsave(file.path(OUTPUT_FIGURES, "green_tilts3.pdf"), width = 8, height = 4, device = "pdf")

if (FALSE) {
  # Check industry score for individual measures
  esg_cols <- c("log_scope1tot", "log_scope12tot", "log_scope123tot", 
              "scope1int_sales", "scope12int_sales", "scope123int_sales", 
              "scope1int_assets", "scope12int_assets", "scope123int_assets", 
              "industry_esg_score", "weighted_esg_score", "e_score", 
              "greenness", "e_climate_score", "e_natres_score", 
              "e_waste_score", "e_envopps_score", "total_esg_score", 
              "environment_score" 
              #"release_total_sales",
              # "prod_waste_total_sales", "release_total_assets", 
              # "prod_waste_total_assets"
            )
  indv <- read_parquet(file.path(DATA_PROCESSED, "df_full.parquet"),
                       col_select = all_of(c("id", "date", esg_cols))) |> setDT()
  indv[, eom := as.Date(date)][, date := NULL]
  indv <- indv[green_scores[, .(id, eom, gics, grp, green_score)], on = .(id, eom)]
  indv <- indv[gics %in% c(55105020, 10102050) & grp %in% c("green", "brown")]
  indv <- indv[, lapply(.SD, mean, na.rm=T), by = .(gics, grp), 
    .SDcols = c("green_score", esg_cols)]
  indv |> 
    pivot_longer(-c(gics, grp)) |> 
    # pivot_wider(names_from = grp, values_from = value) |> 
    # mutate(gmb = green-brown) |>
    mutate(
      ind_pretty = case_when( 
        gics==55105020 ~ paste0("Renewables"),
        gics==10102050 ~ paste0("Coal")
      ),
      grp_pretty = case_when(
        grp == "green" ~ "Green",
        grp == "brown" ~ "Brown"
      ) |> factor(levels = c("Green", "Brown")),
      name = name |> factor(levels = c("green_score", esg_cols))
    ) |> 
    ggplot(aes(name, value, fill = grp_pretty)) + 
    geom_col() +
    facet_wrap(ind_pretty~grp_pretty, ncol = 2) +
    labs(y = "Standardized value") +
    theme(
      legend.position = "none",
      axis.title.x = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  
}