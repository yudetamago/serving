/*
 * metrics_collector_test.cc
 *
 *  Created on: Jun 27, 2017
 *      Author: franck
 */

/* Copyright 2017 Google Inc. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#include "tensorflow_serving/core/metrics_collector.h"

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
namespace tensorflow {
namespace serving {
namespace {

class MetricCollectorTest : MetricCollector {
 public:
  Metric* metric_;
  Status PublishMetric(Metric* metric) {
    metric_ = metric;
    return Status::OK();
  }
};

TEST(MetricsCollectorTest, PublishPredictMetric) {
  string predict_metric_name = "PredictMetric";
  int64 predict_metric_version = 1;
  uint64 predict_time_ms = 10;
  string model_name = "inception";
  int64 model_version = 1;
  bool is_success = true;

  MetricCollectorTest metric_collector;
  MetricCollector::PredictMetric predict_metric = {predict_metric_name, predict_metric_version, predict_time_ms, model_name, model_version, is_success};

  EXPECT_TRUE(metric_collector.PublishMetric(&predict_metric).ok());

  MetricCollector::PredictMetric* result_metric = dynamic_cast<MetricCollector::PredictMetric*>(metric_collector.metric_);
  EXPECT_EQ(predict_metric_name, result_metric->metric_name_);
  EXPECT_EQ(predict_metric_version,result_metric->metric_version_);
  EXPECT_EQ(predict_time_ms, result_metric->predict_time_ms_);
  EXPECT_EQ(model_name, result_metric->model_name_);
  EXPECT_EQ(model_version, result_metric->model_version_);
  EXPECT_EQ(is_success, result_metric->is_success_);

}

TEST(MetricsCollectorTest, PublishPredictMetricSummary) {
  string predict_metric_name = "PredictMetric";
  uint64 prediction_count = 1;
  uint64 predict_time_ms = 10;
  string model_name = "inception";
  int64 model_version = 1;
  uint32 period = 60;
  bool is_success = true;

  MetricCollectorTest metric_collector;
  MetricCollector::PredictMetricSummary predict_metric = {predict_metric_name, prediction_count, predict_time_ms, model_name, model_version, is_success, period};

  EXPECT_TRUE(metric_collector.PublishMetric(&predict_metric).ok());

  MetricCollector::PredictMetricSummary* result_metric = dynamic_cast<MetricCollector::PredictMetricSummary*>(metric_collector.metric_);
  EXPECT_EQ(predict_metric_name, result_metric->metric_name_);
  EXPECT_EQ(prediction_count, result_metric->prediction_count_);
  EXPECT_EQ(predict_time_ms, result_metric->average_predict_time_ms_);
  EXPECT_EQ(model_name, result_metric->model_name_);
  EXPECT_EQ(model_version, result_metric->model_version_);
  EXPECT_EQ(is_success, result_metric->is_success_);
  EXPECT_EQ(period, result_metric->summary_period_);

}

}// namespace
}// namespace serving
}// namespace tensorflow
