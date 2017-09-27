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
#ifndef TENSORFLOW_SERVING_CORE_PREDICT_METRICS_MANAGER_H_
#define TENSORFLOW_SERVING_CORE_PREDICT_METRICS_MANAGER_H_

#include "tensorflow/core/lib/core/status.h"
#include "metrics_manager.h"
#include "metrics_collector.h"
#include "servable_state.h"
#include "servable_state_monitor.h"

namespace tensorflow {
namespace serving {

class PredictMetricsManager : public MetricsManager {
 public:
  virtual ~PredictMetricsManager();
  PredictMetricsManager(MetricCollector& metric_collector,
                        bool enable_metric_summary,
                        int metric_summary_wait_seconds);
  virtual void CreateSummaryMetric(std::map<ServableId, MetricCollector::PredictMetric>& metrics_map, std::map<string, MetricCollector::PredictMetricSummary>& summary_metrics);

};
}
}

#endif /* TENSORFLOW_SERVING_CORE_PREDICT_METRICS_MANAGER_H_ */
