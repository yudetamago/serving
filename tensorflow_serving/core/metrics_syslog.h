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

#ifndef TENSORFLOW_SERVING_CORE_METRICS_SYSLOG_H_
#define TENSORFLOW_SERVING_CORE_METRICS_SYSLOG_H_

#include <syslog.h>
#include "metrics_collector.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace serving {

// Publish metrics on syslog.
class MetricSyslog : public MetricCollector {
 public:
  MetricSyslog() {
    openlog("tensorflow_serving", LOG_PID | LOG_CONS, LOG_USER);
  }
  ;
  ~MetricSyslog() {
    closelog();
  }
  ;

  Status PublishMetric(Metric* metric);

};
}  // namespace serving
}  // namespace tensorflow

#endif /* TENSORFLOW_SERVING_CORE_METRICS_SYSLOG_H_ */
