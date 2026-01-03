import json
import os

folder_path = ''
test_paths = sorted(os.listdir(folder_path))
test_paths = [test_path for test_path in test_paths if os.path.isdir(os.path.join(folder_path, test_path))]
# assert len(test_paths) == 150, "Expected 150 test paths"
sum_metric = {
    'PSNR': 0.0,
    'SSIM': 0.0,
    'LPIPS': 0.0
}
time = 0.0
for test_path in test_paths:
    full_path = os.path.join(folder_path, test_path)
    result_json_path = os.path.join(full_path, f'{test_path}.json')
    with open(result_json_path, 'r') as f:
        result_data = json.load(f)
    avg_metrics = result_data['avg_metric']
    for metric in sum_metric.keys():
        sum_metric[metric] += avg_metrics[metric]
    time += result_data['time']

avg_metric = {metric: sum_metric[metric] / len(test_paths) for metric in sum_metric.keys()}
avg_metric['time'] = time / len(test_paths)
summary_path = os.path.join(folder_path, 'summary.json')
with open(summary_path, 'w') as f:
    json.dump(avg_metric, f, indent=4)
print(avg_metric)
print(f"There are in total {len(test_paths)} test samples.")
print(f"Saved summary metrics to {summary_path}")

