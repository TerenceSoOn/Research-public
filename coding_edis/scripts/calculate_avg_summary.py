import os
import re
import glob
import statistics

def calculate_avg_metrics(base_dir, datasets, m_filter="m16"):
    for dataset in datasets:
        summary_dir = os.path.join(base_dir, dataset, "summary")
        if not os.path.exists(summary_dir):
            print(f"Directory not found: {summary_dir}")
            continue

        file_pattern = os.path.join(summary_dir, f"{m_filter}*_summary.txt")
        files = glob.glob(file_pattern)
        
        if not files:
            print(f"No summary files found for {dataset} with filter {m_filter}")
            continue
            
        print(f"\nProcessing {dataset} ({len(files)} files found):")
        
        metrics = {
            "Avg (all N)": [],
            "Avg_EDIS": [],
            "Best_EDIS": [],
            "Maj_EDIS": []
        }
        
        for file_path in files:
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Extract Avg (all N)
                match = re.search(r"Avg \(all N\)\s+([\d+\.]+)%", content)
                if match:
                    metrics["Avg (all N)"].append(float(match.group(1)))
                    
                # Extract Avg_EDIS@k (using specific @k regex for flexibility)
                match = re.search(r"Avg_EDIS@\d+\s+([\d+\.]+)%", content)
                if match:
                    metrics["Avg_EDIS"].append(float(match.group(1)))

                # Extract Best_EDIS@k
                match = re.search(r"Best_EDIS@\d+\s+([\d+\.]+)%", content)
                if match:
                    metrics["Best_EDIS"].append(float(match.group(1)))
                    
                # Extract Maj_EDIS@k
                match = re.search(r"Maj_EDIS@\d+\s+([\d+\.]+)%", content)
                if match:
                    metrics["Maj_EDIS"].append(float(match.group(1)))

        # Calculate and print averages
        print("-" * 40)
        print(f"{'Metric':<20} {'Average Accuracy':>15}")
        print("-" * 40)
        
        for key, values in metrics.items():
            if values:
                avg_val = statistics.mean(values)
                print(f"{key:<20} {avg_val:>14.1f}%")
            else:
                print(f"{key:<20} {'N/A':>15}")
        print("-" * 40)

if __name__ == "__main__":
    base_directory = "/Users/terence/Desktop/EDIS_trial/run_result"
    target_datasets = ["aime24", "amc23", "math500"]
    
    calculate_avg_metrics(base_directory, target_datasets, m_filter="m16")
