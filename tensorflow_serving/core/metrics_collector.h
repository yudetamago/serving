/* Copyright 2016 Google Inc. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/

#ifndef TENSORFLOW_SERVING_CORE_METRICS_COLLECTOR_H_
#define TENSORFLOW_SERVING_CORE_METRICS_COLLECTOR_H_

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace serving {

// MetricCollector defines an abstract interface to use for publishing metrics.
class MetricCollector {
 public:
  // Abstract structure for metric.
  struct Metric {
    Metric() = default;
    Metric(const string metric_name, const int64 metric_version,
           const string model_name, const int64 model_version)
        : metric_name_(metric_name),
          metric_version_(metric_version),
          model_name_(model_name),
          model_version_(model_version) {
    }

    // Returns a string representation of this object. Useful in logging.
    virtual string DebugString() = 0;
    virtual ~Metric() {}
    string metric_name_;
    int64 metric_version_ = 0;

    string model_name_;
    int64 model_version_ = 0;
  };

  // Structure for predict metric
  struct PredictMetric : public Metric {
    PredictMetric() = default;
    PredictMetric(const string metric_name, const int64 metric_version,
                  const uint64 predict_time_ms, const string model_name,
                  const int64 model_version, const bool is_success)
        : Metric(metric_name, metric_version, model_name, model_version),
          predict_time_ms_(predict_time_ms),
          is_success_(is_success) {
    }

    // Returns a string representation of this object. Useful in logging.
    string DebugString() {
      return strings::StrCat("metric_name=\"", metric_name_, "\"",
                             " metric_version=\"",
                             std::to_string(metric_version_), "\"",
                             " model_name=\"", model_name_, "\"",
                             " model_version=\"",
                             std::to_string(model_version_), "\"",
                             " is_success=\"", std::to_string(is_success_),
                             "\"", " predict_time_ms=\"",
                             std::to_string(predict_time_ms_), "\"");
    }
    virtual ~PredictMetric(){}
    uint64 predict_time_ms_ = 0;
    bool is_success_ = false;
  };

  // Structure for summary predict metric
  struct PredictMetricSummary : public Metric {
    PredictMetricSummary() = default;
    PredictMetricSummary(const string metric_name,
                         const uint64 prediction_count,
                         const uint64 average_predict_time_ms,
                         const string model_name, const int64 model_version,
                         const bool is_success, const uint32 summary_period)
        : Metric(metric_name, 0, model_name, model_version),
          prediction_count_(prediction_count),
          summary_period_(summary_period),
          average_predict_time_ms_(average_predict_time_ms),
          is_success_(is_success) {
    }
    // Returns a string representation of this object. Useful in logging.
    string DebugString() {
      return strings::StrCat("metric_name=\"", metric_name_, "\"",
                             " prediction_count=\"",
                             std::to_string(prediction_count_), "\"",
                             " model_name=\"", model_name_, "\"",
                             " model_version=\"",
                             std::to_string(model_version_), "\"",
                             " is_success=\"", std::to_string(is_success_),
                             "\"", " average_predict_time_ms=\"",
                             std::to_string(average_predict_time_ms_), "\"",
                             " summary_period=\"",
                             std::to_string(summary_period_), "\"");
    }

    uint64 prediction_count_ = 0;
    uint32 summary_period_ = 30;
    uint64 average_predict_time_ms_ = 0;
    bool is_success_ = false;

  };
  virtual ~MetricCollector() = default;
  virtual Status PublishMetric(Metric* metric) = 0;

 protected:
  MetricCollector() = default;
};
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_METRICS_COLLECTOR_H_
