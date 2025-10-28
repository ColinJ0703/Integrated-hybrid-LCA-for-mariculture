

temp_dir <- "D:/Temp"
if (!dir.exists(temp_dir)) dir.create(temp_dir, recursive = TRUE)
Sys.setenv(TMPDIR = temp_dir)


pkgs <- c("readxl", "writexl", "networkD3", "htmlwidgets",
          "webshot", "htmltools", "rstudioapi",
          "magick", "magrittr", "jsonlite")
for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
lapply(pkgs, library, character.only = TRUE)
if (!webshot::is_phantomjs_installed()) webshot::install_phantomjs()


output_dir <- tryCatch(dirname(rstudioapi::getActiveDocumentContext()$path),
                       error = function(e) getwd())
setwd(output_dir)
file_path <- file.path(output_dir, "Path.xlsx")
if (!file.exists(file_path)) stop("No Path.xlsx")


canon <- function(x) tolower(trimws(x))
sheets <- c("Fish", "Shrimp", "Shellfish", "Algae")


dfs <- list(); all_nodes <- start_nodes <- character(0)
for (sh in sheets) {
  df <- read_excel(file_path, sheet = sh, col_names = TRUE)
  dfs[[sh]] <- df
  all_nodes  <- c(all_nodes,
                  unlist(df[, 1:(ncol(df)-1)], use.names = FALSE) |> na.omit())
  start_nodes <- c(start_nodes,
                   df[[1]][which(!is.na(df[[1]]))[1]])
}

canon_all   <- canon(all_nodes)
canon_start <- canon(start_nodes)
ordered_nodes_canon <- c(canon_start,
                         sort(setdiff(unique(canon_all), canon_start)))


fixed_colors <- c(
  "fish culture"      = "#3183BE",
  "shrimp culture"    = "#3183BE",
  "shellfish culture" = "#3183BE",
  "algae culture"     = "#3183BE",
  "carbon footprint"  = "#D98880"
)

gradient_stops <- c(
  "#C6DCF0", "#9ECBE2", "#6BADD7",
  "#BEBEBE",
  "#5EC2B7", "#2D8C4A",
  "#FFF2CC", "#FFD966", "#F4B183", "#FEC000",
  "#C61B8B", "#7B0178"
)


n_other <- sum(!ordered_nodes_canon %in% names(fixed_colors))
palette_other <- colorRampPalette(gradient_stops)(max(n_other, length(gradient_stops)))


color_map <- setNames(character(length(ordered_nodes_canon)),
                      ordered_nodes_canon)
idx <- 1
for (c in ordered_nodes_canon) {
  color_map[c] <- if (c %in% names(fixed_colors)) fixed_colors[c]
  else { col <- palette_other[idx]; idx <- idx + 1; col }
}


canon_to_orig <- tapply(all_nodes, canon_all, `[`, 1)
sheet_presence <- lapply(dfs, function(df)
  canon(unlist(df[, 1:(ncol(df)-1)], use.names = FALSE)))
names(sheet_presence) <- sheets
present <- \(sh) ifelse(ordered_nodes_canon %in% sheet_presence[[sh]], "✔", "")
mapping_df <- data.frame(
  Node       = canon_to_orig[ordered_nodes_canon],
  Color      = unname(color_map[ordered_nodes_canon]),
  Fish       = present("Fish"),
  Shrimp     = present("Shrimp"),
  Shellfish  = present("Shellfish"),
  Algae      = present("Algae"),
  check.names = FALSE
)
writexl::write_xlsx(mapping_df, "node_color_mapping.xlsx")


unique_cols <- unique(unname(color_map))
js_col <- sprintf(
  "d3.scaleOrdinal().domain(%s).range(%s)",
  jsonlite::toJSON(unique_cols),
  jsonlite::toJSON(unique_cols)
)


draw_sankey <- function(sheet, labeled = TRUE) {
  df <- dfs[[sheet]]
  links <- data.frame()
  for (i in seq_len(nrow(df))) {
    path <- na.omit(unlist(df[i, 1:(ncol(df)-1)], use.names = FALSE))
    val  <- as.numeric(df[i, ncol(df)])
    if (length(path) > 1 && !is.na(val) && val != 0)
      links <- rbind(links,
                     data.frame(source = head(path, -1),
                                target = tail(path, -1),
                                value  = val))
  }
  nodes <- data.frame(name = unique(c(links$source, links$target)))
  nodes$canon  <- canon(nodes$name)
  nodes$colour <- unname(color_map[nodes$canon])
  nodes$label  <- if (labeled) nodes$name else ""
  links$source <- match(links$source, nodes$name) - 1
  links$target <- match(links$target, nodes$name) - 1
  links$colour <- nodes$colour[links$target + 1]
  
  sankeyNetwork(
    Links      = links, Nodes = nodes,
    Source     = "source", Target = "target",
    Value      = "value",  NodeID = "label",
    NodeGroup  = "colour", LinkGroup = "colour",
    fontSize   = 8, nodeWidth = 18,
    colourScale = JS(js_col),
    height = 380, width = 720
  )
}


for (sh in sheets) {
  for (tag in c("_label","_nolabel")) {
    fig <- draw_sankey(sh, labeled = tag == "_label")
    html <- tempfile(fileext = ".html")
    saveNetwork(fig, html, selfcontained = TRUE)
    png  <- sprintf("sankey_%s%s.png", sh, tag)
    webshot(html, png, vwidth = 800, vheight = 400, zoom = 3)
    unlink(html)
  }
}


imgs <- lapply(sheets, function(sh)
  image_read(sprintf("sankey_%s_nolabel.png", sh)) |>
    image_trim() |> image_scale("720x") |> image_border("transparent", "40x40"))
grid <- image_append(
  c(image_append(c(imgs[[1]], imgs[[2]])),
    image_append(c(imgs[[3]], imgs[[4]]))),
  stack = TRUE)
image_write(grid, "sankey_combined_nolabel.png")
message("Finish: node_color_mapping.xlsx + PNG")
