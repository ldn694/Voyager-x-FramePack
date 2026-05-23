import json
import os
import matplotlib.pyplot as plt

folder_path = 'evaluation/RealEstate10K/refined_test_150_768x512_all-4x4_no-cfg_flexdit-loss'
test_paths = sorted(os.listdir(folder_path))
test_paths = [test_path for test_path in test_paths if os.path.isdir(os.path.join(folder_path, test_path))]
# assert len(test_paths) == 150, "Expected 150 test paths"
sum_metric = {
    'PSNR': 0.0,
    'SSIM': 0.0,
    'LPIPS': 0.0
}
time = 0.0
diffusion_time = 0.0
cnt = 0

# Lists to collect data for the bar chart
sample_names = []
sample_times = []

for test_path in test_paths:
    full_path = os.path.join(folder_path, test_path)
    result_json_path = os.path.join(full_path, f'{test_path}.json')
    if not os.path.exists(result_json_path):
        continue
    with open(result_json_path, 'r') as f:
        # print(f)
        result_data = json.load(f)
    avg_metrics = result_data['avg_metric']
    for metric in sum_metric.keys():
        sum_metric[metric] += avg_metrics[metric]
    current_time = result_data['time']
    time += current_time
    cnt += 1
    if 'diffusion_time' in result_data.keys():
        diffusion_time += result_data['diffusion_time']
    
    # Store the individual test time and name
    sample_names.append(test_path)
    sample_times.append(current_time)

avg_metric = {metric: sum_metric[metric] / len(test_paths) for metric in sum_metric.keys()}
avg_metric['time'] = time / cnt
if diffusion_time > 0:
    avg_metric['diffusion_time'] = diffusion_time / cnt
summary_path = os.path.join(folder_path, 'summary.json')
with open(summary_path, 'w') as f:
    json.dump(avg_metric, f, indent=4)
print(avg_metric)
print(f"There are in total {cnt} test samples.")
print(f"Saved summary metrics to {summary_path}")


# --- Histogram Generation ---
plt.figure(figsize=(10, 6))

# Create the histogram
# bins='auto' automatically decides the number of bins based on data distribution
plt.hist(sample_times, bins='auto', color='skyblue', edgecolor='black', alpha=0.7)

plt.xlabel('Total Time (s)', fontsize=12)
plt.ylabel('Count (Number of Samples)', fontsize=12)
plt.title('Distribution of Execution Time', fontsize=14, pad=15)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

chart_path = os.path.join(folder_path, 'time_distribution_histogram.png')
plt.savefig(chart_path, dpi=300)
print(f"Saved time distribution histogram to {chart_path}")
plt.close()

