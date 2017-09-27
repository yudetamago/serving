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
#include <chrono>
#include <thread>
#include <future>

#include "predict_metrics_manager.h"
#include "metrics_logger.h"
#include "metrics_syslog.h"

using ServableStateAndTime = tensorflow::serving::ServableStateMonitor::ServableStateAndTime;

namespace tensorflow {
namespace serving {

std::map<ServableId, MetricCollector::PredictMetric> metrics_map_;
std::map<string, MetricCollector::PredictMetricSummary> summary_metrics_;
uint32 metric_summary_wait_seconds_;
MetricCollector* metric_collector_;
std::future<void> summary_async_;
std::atomic<bool> summary_thread_running_ { true };

PredictMetricsManager::PredictMetricsManager(MetricCollector& metric_collector,
                                             bool enable_metric_summary,
                                             int metric_summary_wait_seconds) {
  metric_collector_ = &metric_collector;
  metric_summary_wait_seconds_ = metric_summary_wait_seconds;
  if (enable_metric_summary) {
    // Start an async task for publishing summary metrics every $metric_summary_wait_seconds seconds.
    summary_async_ = std::async(std::launch::async, [this] {
      while(summary_thread_running_)
      {
        this->CreateSummaryMetric(metrics_map_, summary_metrics_);
        for (auto &metric : summary_metrics_) {
          metric_collector_->PublishMetric(&metric.second);
        }

        metrics_map_.clear();
        summary_metrics_.clear();
        std::this_thread::sleep_for(
            std::chrono::seconds(metric_summary_wait_seconds_));
      }
    });
  }
}

void PredictMetricsManager::CreateSummaryMetric(
    std::map<ServableId, MetricCollector::PredictMetric>& metrics_map,
    std::map<string, MetricCollector::PredictMetricSummary>& summary_metrics) {

  uint64 success_predict_count = 0;
  uint64 failed_predict_count = 0;
  uint64 predict_count = 0;
  for (auto &metrics : metrics_map) {
    string success_summary_key = metrics.second.model_name_
        + std::to_string(metrics.second.model_version_) + "_success";
    string failed_summary_key = metrics.second.model_name_
        + std::to_string(metrics.second.model_version_) + "_failed";
    string summary_key;
    if (metrics.second.is_success_) {
      summary_key = success_summary_key;
      predict_count = ++success_predict_count;
    } else {
      summary_key = failed_summary_key;
      predict_count = ++failed_predict_count;
    }

    if (summary_metrics.find(summary_key) != summary_metrics.end()) {
      MetricCollector::PredictMetricSummary& metric =
          summary_metrics[summary_key];
      uint64 average_predict_time_ms = (metric.average_predict_time_ms_
          + metrics.second.predict_time_ms_) / predict_count;
      summary_metrics[summary_key] = {"PredictSummary", predict_count, average_predict_time_ms, metrics.second.model_name_, metrics.second.model_version_, metrics.second.is_success_, metric_summary_wait_seconds_};
    } else {
      summary_metrics[summary_key] = {"PredictSummary", 1, metrics.second.predict_time_ms_, metrics.second.model_name_, metrics.second.model_version_, metrics.second.is_success_, metric_summary_wait_seconds_};
    }
  }

}

PredictMetricsManager::~PredictMetricsManager() {
  KillSummaryThread();
}

ServableStateMonitor::ServableStateNotifierFn MetricsManager::CreateNotifier(
    const uint64& elapsed_predict_time, const Status& result_status,
    const string& model_name, const int64& model_version) {

  const ServableStateMonitor::ServableStateNotifierFn& notifier_fn =
      [&](const bool reached,
          std::map<ServableId, ServableState::ManagerState> states_reached) {
        for(auto state_reached : states_reached) {
          switch(state_reached.second) {
            case ServableState::ManagerState::kStart :
            case ServableState::ManagerState::kUnloading :
            case ServableState::ManagerState::kAvailable :
            case ServableState::ManagerState::kLoading :
            break;
            // We listen only end event for metrics.
            case ServableState::ManagerState::kEnd :
            auto elapsed_predict_time_ms = elapsed_predict_time / 1000;

            // Build a predict metric.
            MetricCollector::PredictMetric metric = {state_reached.first.name, state_reached.first.version, elapsed_predict_time_ms, model_name, model_version, result_status.ok()};

            // Store the metric on a map needed for the summary.
            metrics_map_[state_reached.first] = metric;

            // Publish metric.
            metric_collector_->PublishMetric(&metric);
            break;
          }
        }
      };
  return notifier_fn;
}

Status MetricsManager::KillSummaryThread() {
  summary_thread_running_ = false;
  return Status::OK();
}

// Create metrics publisher and instantiate the metrics manager.
Status MetricsManager::Create(
    const string target, const bool enable_metric_summary,
    const int32 metric_summary_wait_seconds,
    std::unique_ptr<MetricsManager>* metrics_manager) {
  MetricCollector* metric_collector;
  if (target == "logger") {
    metric_collector = new MetricLogger();
  } else if (target == "syslog") {
    metric_collector = new MetricSyslog();
  }
  metrics_manager->reset(
      new PredictMetricsManager(*metric_collector, enable_metric_summary,
                                metric_summary_wait_seconds));
  return Status::OK();
}

}
}

