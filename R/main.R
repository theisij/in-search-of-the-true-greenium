# Packages -----------------------------------------------------------------------------
library(arrow)
library(lfe)
library(broom)
library(tidyverse) 
library(data.table)

# Layout settings ----------------------------------------------------------------------
theme_set(theme_classic())
colours_theme <- c("#008000",  # Green in matplotlib
                   "#FFA500",  # Orange in matplotlib
                   "#0C6291", "#A63446", RColorBrewer::brewer.pal(8, "Dark2"), 
                   "darkslategrey", "blue3", "red3", "purple2", "yellow2", "aquamarine",
                   "grey", "salmon", "antiquewhite", "chartreuse") 
scale_colour_discrete <- function(...) {
  scale_colour_manual(..., values = colours_theme)
}
scale_fill_discrete <- function(...) {
  scale_fill_manual(..., values = colours_theme)
}

# Configurations -----------------------------------------------------------------------
# PATHS
DATA <- file.path("data")
DATA_RAW <- file.path(DATA, "raw")
DATA_INTERIM <- file.path(DATA, "interim")
DATA_PROCESSED <- file.path(DATA, "processed")
OUTPUT <- file.path("output")
OUTPUT_FIGURES <- file.path(OUTPUT, "figures")

# Run code -----------------------------------------------------------------------------
source("R/standard_errors.R", echo = T)
source("R/green_tilts.R", echo = T)
source("R/random_gs.R", echo = T)