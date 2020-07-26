library(lintr)
library(crayon)

my_linters <- list(
  absolute_path_linter = lintr::absolute_path_linter,
  assignment_linter = lintr::assignment_linter,
  closed_curly_linter = lintr::closed_curly_linter,
  commas_linter = lintr::commas_linter,
  # commented_code_linter = lintr::commented_code_linter,
  infix_spaces_linter = lintr::infix_spaces_linter,
  line_length_linter = lintr::line_length_linter,
  no_tab_linter = lintr::no_tab_linter,
  object_usage_linter = lintr::object_usage_linter,
  # snake_case_linter = lintr::snake_case_linter,
  # multiple_dots_linter = lintr::multiple_dots_linter,
  object_length_linter = lintr::object_length_linter,
  open_curly_linter = lintr::open_curly_linter,
  # single_quotes_linter = lintr::single_quotes_linter,
  spaces_inside_linter = lintr::spaces_inside_linter,
  spaces_left_parentheses_linter = lintr::spaces_left_parentheses_linter,
  trailing_blank_lines_linter = lintr::trailing_blank_lines_linter,
  trailing_whitespace_linter = lintr::trailing_whitespace_linter,
  true_false = lintr::T_and_F_symbol_linter
)

results <- lapply(
  list.files(path = '.', pattern = '\\.[Rr]$', recursive = TRUE),
  function (r_file) {
    cat(sprintf("Processing %s ...\n", r_file))
    list(r_file = r_file,
         output = lintr::lint(filename = r_file, linters = my_linters))
  })
num_issue <- Reduce(sum, lapply(results, function (e) length(e$output)))

lint2str <- function(lint_entry) {
  color <- function(type) {
    switch(type,
      "warning" = crayon::magenta,
      "error" = crayon::red,
      "style" = crayon::blue,
      crayon::bold
    )
  }

  paste0(
    lapply(lint_entry$output,
      function (lint_line) {
        paste0(
          crayon::bold(lint_entry$r_file, ":",
          as.character(lint_line$line_number), ":",
          as.character(lint_line$column_number), ": ", sep = ""),
          color(lint_line$type)(lint_line$type, ": ", sep = ""),
          crayon::bold(lint_line$message), "\n",
          lint_line$line, "\n",
          lintr:::highlight_string(lint_line$message, lint_line$column_number, lint_line$ranges),
          "\n",
          collapse = "")
      }),
    collapse = "")
}

if (num_issue > 0) {
  cat(sprintf('R linters found %d issues:\n', num_issue))
  for (entry in results) {
    if (length(entry$output)) {
      cat(paste0('**** ', crayon::bold(entry$r_file), '\n'))
      cat(paste0(lint2str(entry), collapse = ''))
    }
  }
  quit(save = 'no', status = 1)  # Signal error to parent shell
}
