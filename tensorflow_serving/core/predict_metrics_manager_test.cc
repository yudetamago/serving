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
 ==============================================================================*/

#include "tensorflow_serving/core/predict_metrics_manager.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "tensorflow/contrib/batching/test_util/fake_clock_env.h"
#include "metrics_logger.h"
#include "metrics_collector.h"

namespace tensorflow {
namespace serving {
namespace {
using ::testing::_;
using ::testing::HasSubstr;
using ::testing::Invoke;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::KilledBySignal;

class MockMetricCollector : public MetricCollector {
 public:
  MOCK_METHOD1(PublishMetric, Status(MetricCollector::Metric* metric));
};

using ServableStateMonitorCreator =
std::function<Status(EventBus<ServableState>* event_bus,
    std::unique_ptr<ServableStateMonitor>* monitor)>;

TEST(PredictMetricsManagerTest, CreateNotifier) {
  uint64 event_start_time = 10;
  string model_name = "inception";
  int64 model_version = 1;
  Status result_status = Status::OK();
  string predict_metric_name = "PredictMetric";
  int64 predict_metric_version = 1;
  ServableState::ManagerState manager_state = ServableState::ManagerState::kEnd;

  test_util::FakeClockEnv env(Env::Default());
  EventBus<ServableState>::Options bus_options;
  bus_options.env = &env;
  auto bus = EventBus<ServableState>::CreateEventBus(bus_options);

  ServableStateMonitor::Options monitor_options;
  monitor_options.max_count_log_events = 4;

  ServableStateMonitor monitor(bus.get(), monitor_options);

  MockMetricCollector* metric_collector_ = new MockMetricCollector();
  PredictMetricsManager* manager = new PredictMetricsManager(*metric_collector_,
                                                             false, 1);
  ServableStateMonitor::ServableStateNotifierFn notifier_fn = manager
      ->CreateNotifier(event_start_time, result_status, model_name,
                       model_version);

  std::vector<ServableRequest> servables;
  const ServableId specific_goal_state_id = { predict_metric_name,
      predict_metric_version };
  servables.push_back(ServableRequest::FromId(specific_goal_state_id));
  monitor.NotifyWhenServablesReachState(servables, manager_state, notifier_fn);

  ServableState event = { ServableId { predict_metric_name,
      predict_metric_version }, manager_state, Status::OK() };

  EXPECT_CALL(*metric_collector_, PublishMetric(_)).WillOnce(
      Return(Status::OK()));
  EXPECT_NE(nullptr, notifier_fn);

  bus.get()->Publish(event);
  metric_collector_->~MockMetricCollector();

}

TEST(PredictMetricsManagerTest, CreateSummaryMetric) {
  MockMetricCollector* metric_collector_ = new MockMetricCollector();
  PredictMetricsManager* manager = new PredictMetricsManager(*metric_collector_,
                                                             false, 1);
  string predict_metric_name = "PredictMetric";

  MetricCollector::PredictMetric metric = { predict_metric_name, 1, 10,
      "inception", 1, true };
  MetricCollector::PredictMetric metric2 = { predict_metric_name, 2, 20,
      "inception", 1, true };
  MetricCollector::PredictMetric metric3 = { predict_metric_name, 3, 5,
      "inception", 2, true };
  MetricCollector::PredictMetric metric4 = { predict_metric_name, 4, 5,
      "resnet", 1, true };
  MetricCollector::PredictMetric metric5 = { predict_metric_name, 4, 5,
      "resnet", 1, false };
  const ServableId specific_goal_state_id = { predict_metric_name, 1 };
  const ServableId specific_goal_state_id2 = { predict_metric_name, 2 };
  const ServableId specific_goal_state_id3 = { predict_metric_name, 3 };
  const ServableId specific_goal_state_id4 = { predict_metric_name, 4 };
  const ServableId specific_goal_state_id5 = { predict_metric_name, 5 };

  std::map<ServableId, MetricCollector::PredictMetric> metrics_map = { {
      specific_goal_state_id, metric }, { specific_goal_state_id2, metric2 }, {
      specific_goal_state_id3, metric3 }, { specific_goal_state_id4, metric4 },
      { specific_goal_state_id5, metric5 } };
  std::map<string, MetricCollector::PredictMetricSummary> summary_metrics;
  manager->CreateSummaryMetric(metrics_map, summary_metrics);

  EXPECT_EQ(4, summary_metrics.size());
  EXPECT_EQ(true, summary_metrics["inception1_success"].is_success_);
  EXPECT_EQ(15, summary_metrics["inception1_success"].average_predict_time_ms_);
  EXPECT_EQ("PredictSummary",
            summary_metrics["inception1_success"].metric_name_);
  EXPECT_EQ(0, summary_metrics["inception1_success"].metric_version_);
  EXPECT_EQ("inception", summary_metrics["inception1_success"].model_name_);
  EXPECT_EQ(1, summary_metrics["inception1_success"].model_version_);
  EXPECT_EQ(2, summary_metrics["inception1_success"].prediction_count_);
  EXPECT_EQ(1, summary_metrics["inception1_success"].summary_period_);

  EXPECT_EQ(true, summary_metrics["inception2_success"].is_success_);
  EXPECT_EQ(5, summary_metrics["inception2_success"].average_predict_time_ms_);
  EXPECT_EQ("PredictSummary",
            summary_metrics["inception2_success"].metric_name_);
  EXPECT_EQ(0, summary_metrics["inception2_success"].metric_version_);
  EXPECT_EQ("inception", summary_metrics["inception2_success"].model_name_);
  EXPECT_EQ(2, summary_metrics["inception2_success"].model_version_);
  EXPECT_EQ(1, summary_metrics["inception2_success"].prediction_count_);
  EXPECT_EQ(1, summary_metrics["inception2_success"].summary_period_);

  EXPECT_EQ(true, summary_metrics["resnet1_success"].is_success_);
  EXPECT_EQ(5, summary_metrics["resnet1_success"].average_predict_time_ms_);
  EXPECT_EQ("PredictSummary", summary_metrics["resnet1_success"].metric_name_);
  EXPECT_EQ(0, summary_metrics["resnet1_success"].metric_version_);
  EXPECT_EQ("resnet", summary_metrics["resnet1_success"].model_name_);
  EXPECT_EQ(1, summary_metrics["resnet1_success"].model_version_);
  EXPECT_EQ(1, summary_metrics["resnet1_success"].prediction_count_);
  EXPECT_EQ(1, summary_metrics["resnet1_success"].summary_period_);

  EXPECT_EQ(false, summary_metrics["resnet1_failed"].is_success_);
  EXPECT_EQ(5, summary_metrics["resnet1_failed"].average_predict_time_ms_);
  EXPECT_EQ("PredictSummary", summary_metrics["resnet1_failed"].metric_name_);
  EXPECT_EQ(0, summary_metrics["resnet1_failed"].metric_version_);
  EXPECT_EQ("resnet", summary_metrics["resnet1_failed"].model_name_);
  EXPECT_EQ(1, summary_metrics["resnet1_failed"].model_version_);
  EXPECT_EQ(1, summary_metrics["resnet1_failed"].prediction_count_);
  EXPECT_EQ(1, summary_metrics["resnet1_failed"].summary_period_);

  metric_collector_->~MockMetricCollector();

}

TEST(MetricsFactoryTest, CreateWithLogger) {
  std::unique_ptr<MetricsManager> metrics_manager;
  Status status = MetricsManager::Create("logger", false, 30, &metrics_manager);
  EXPECT_TRUE(status.ok());
  EXPECT_NE(nullptr, metrics_manager.get());

}

TEST(MetricsFactoryTest, CreateWithSyslog) {
  std::unique_ptr<MetricsManager> metrics_manager;
  Status status = MetricsManager::Create("syslog", false, 30, &metrics_manager);
  EXPECT_TRUE(status.ok());
  EXPECT_NE(nullptr, metrics_manager.get());

}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
