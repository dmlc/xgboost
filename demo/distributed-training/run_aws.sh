# This is the example script to run distributed xgboost on AWS.
# Change the following two lines for configuration

export BUCKET=mybucket

# submit the job to YARN
../../dmlc-core/tracker/dmlc-submit --cluster=yarn --num-workers=2 --worker-cores=2\
    ../../xgboost mushroom.aws.conf nthread=2\
    data=s3://${BUCKET}/xgb-demo/train\
    eval[test]=s3://${BUCKET}/xgb-demo/test\
    model_dir=s3://${BUCKET}/xgb-demo/model
