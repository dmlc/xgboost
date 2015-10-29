context("Code is of high quality and lint free")
test_that("Code Lint", {
  skip_on_cran()
  skip_on_travis()
  skip_if_not_installed("lintr")
  my_linters <- list(
    absolute_paths_linter=lintr::absolute_paths_linter,
    assignment_linter=lintr::assignment_linter,
    closed_curly_linter=lintr::closed_curly_linter,
    commas_linter=lintr::commas_linter,
    # commented_code_linter=lintr::commented_code_linter,
    infix_spaces_linter=lintr::infix_spaces_linter,
    line_length_linter=lintr::line_length_linter,
    no_tab_linter=lintr::no_tab_linter,
    object_usage_linter=lintr::object_usage_linter,
    # snake_case_linter=lintr::snake_case_linter,
    # multiple_dots_linter=lintr::multiple_dots_linter,
    object_length_linter=lintr::object_length_linter,
    open_curly_linter=lintr::open_curly_linter,
    # single_quotes_linter=lintr::single_quotes_linter,
    spaces_inside_linter=lintr::spaces_inside_linter,
    spaces_left_parentheses_linter=lintr::spaces_left_parentheses_linter,
    trailing_blank_lines_linter=lintr::trailing_blank_lines_linter,
    trailing_whitespace_linter=lintr::trailing_whitespace_linter
  )
  # lintr::expect_lint_free(linters=my_linters) # uncomment this if you want to check code quality
})
