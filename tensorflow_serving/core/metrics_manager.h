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
#ifndef TENSORFLOW_SERVING_CORE_METRICS_MANAGER_H_
#define TENSORFLOW_SERVING_CORE_METRICS_MANAGER_H_

#include <atomic>
#include "tensorflow/core/lib/core/status.h"
#include "servable_state.h"
#include "servable_state_monitor.h"

namespace tensorflow {
namespace serving {

class MetricsManager {

 public:

  // Create metrics publisher and instantiate the metrics manager.
  static Status Create(const string target, const bool enable_metric_summary,
                       const int32 metric_summary_wait_seconds,
                       std::unique_ptr<MetricsManager>* metrics_manager);

  // Create a notifier to handle metrics end event.
  ServableStateMonitor::ServableStateNotifierFn CreateNotifier(
      const uint64& elapsed_predict_time, const Status& result_status,
      const string& model_name, const int64& model_version);

  Status KillSummaryThread();

};
}  // namespace serving
}  // namespace tensorflow

#endif /* TENSORFLOW_SERVING_CORE_METRICS_MANAGER_H_ */
