## A special file sourced by testthat.

get_basescore <- function(model) {
  as.numeric(
    jsonlite::fromJSON(model$learner$learner_model_param$base_score)
  )
}
