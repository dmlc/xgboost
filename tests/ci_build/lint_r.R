library(lintr)

args <- commandArgs(
    trailingOnly = TRUE
)
SOURCE_DIR <- args[[1L]]

FILES_TO_LINT <- list.files(
    path = SOURCE_DIR
    , pattern = "\\.r$|\\.rmd$"
    , all.files = TRUE
    , ignore.case = TRUE
    , full.names = TRUE
    , recursive = TRUE
    , include.dirs = FALSE
)

my_linters <- list(
  absolute_path_linter = lintr::absolute_path_linter(),
  any_duplicated = lintr::any_duplicated_linter(),
  any_is_na = lintr::any_is_na_linter(),
  assignment_linter = lintr::assignment_linter(),
  brace_linter = lintr::brace_linter(),
  commas_linter = lintr::commas_linter(),
  equals_na = lintr::equals_na_linter(),
  fixed_regex = lintr::fixed_regex_linter(),
  infix_spaces_linter = lintr::infix_spaces_linter(),
  line_length_linter = lintr::line_length_linter(length = 150L),
  no_tab_linter = lintr::no_tab_linter(),
  object_usage_linter = lintr::object_usage_linter(),
  object_length_linter = lintr::object_length_linter(),
  semicolon = lintr::semicolon_linter(),
  seq = lintr::seq_linter(),
  spaces_inside_linter = lintr::spaces_inside_linter(),
  spaces_left_parentheses_linter = lintr::spaces_left_parentheses_linter(),
  sprintf = lintr::sprintf_linter(),
  trailing_blank_lines_linter = lintr::trailing_blank_lines_linter(),
  trailing_whitespace_linter = lintr::trailing_whitespace_linter(),
  true_false = lintr::T_and_F_symbol_linter(),
  unneeded_concatenation = lintr::unneeded_concatenation_linter(),
  unreachable_code = lintr::unreachable_code_linter(),
  vector_logic = lintr::vector_logic_linter()
)

noquote(paste0(length(FILES_TO_LINT), " R files need linting"))

results <- NULL

for (r_file in FILES_TO_LINT) {

    this_result <- lintr::lint(
        filename = r_file
        , linters = my_linters
        , cache = FALSE
    )

    print(
        sprintf(
            "Found %i linting errors in %s"
            , length(this_result)
            , r_file
        )
        , quote = FALSE
    )

    results <- c(results, this_result)

}

issues_found <- length(results)

noquote(paste0("Total linting issues found: ", issues_found))

if (issues_found > 0L) {
    print(results)
    quit(save = "no", status = 1L)
}
