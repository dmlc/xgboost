package com.airbnb.common.ml.xgboost.data

import ml.dmlc.xgboost4j.LabeledPoint

class ScoringLabeledPoint(val data: String, val labeledPoint: LabeledPoint) extends Serializable



