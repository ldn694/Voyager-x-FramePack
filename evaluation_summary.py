import os
import json

def get_sub_folders(root):
    return sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))])

root_path = 'evaluation/RealEstate10K'
test_folders = get_sub_folders(root_path)

exlude_list = ['refined_test_150_768x512']
test_folders = [test_folder for test_folder in test_folders if test_folder not in exlude_list]

tests = None

for test_folder in test_folders:
    current_tests = get_sub_folders(os.path.join(root_path, test_folder))
    assert len(current_tests) == 150, f"Expected 150 test paths in {test_folder}"
    if tests is None:
        tests = current_tests
    else:
        assert tests == current_tests, f"Test paths do not match in {test_folder}"

per_test_scores = {test: {'PSNR': 0.0, 'SSIM': 0.0, 'LPIPS': 0.0} for test in tests}

for test_folder in test_folders:
    metrics = {'PSNR': [], 'SSIM': [], 'LPIPS': []}
    for test in tests:
        result_json_path = os.path.join(root_path, test_folder, test, f'{test}.json')
        with open(result_json_path, 'r') as f:
            result_data = json.load(f)
        avg_metrics = result_data['avg_metric']
        for metric in metrics.keys():
            metrics[metric].append(avg_metrics[metric])
    
    for metric in metrics.keys():
        if metric == 'LPIPS':
            metrics[metric] = sorted(metrics[metric], reverse=True)
        else:
            metrics[metric] = sorted(metrics[metric])
        # first is best
    
    for test in tests:
        result_json_path = os.path.join(root_path, test_folder, test, f'{test}.json')
        with open(result_json_path, 'r') as f:
            result_data = json.load(f)
        avg_metrics = result_data['avg_metric']
        for metric in metrics.keys():
            sign = 1 if metric == 'LPIPS' else -1
            l = 0
            r = len(metrics[metric]) - 1
            while l <= r:
                mid = (l + r) // 2
                # if metrics[metric][mid] < avg_metrics[metric]:
                if sign * metrics[metric][mid] > sign * avg_metrics[metric]:
                    l = mid + 1
                else:
                    r = mid - 1
            per_test_scores[test][metric] += l

for test in tests:
    per_test_scores[test]['total'] = sum(per_test_scores[test][metric] for metric in ['PSNR', 'SSIM', 'LPIPS'])

per_test_scores = dict(sorted(per_test_scores.items(), key=lambda item: item[1]['total']))

summary_path = os.path.join(root_path, 'summary_realestate10K.json')
with open(summary_path, 'w') as f:
    json.dump(per_test_scores, f, indent=4)
print(f"Saved per-test summary metrics to {summary_path}")

    