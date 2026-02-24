"""
Script to analyze DEAP pickle files and calculate true F1 scores
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pandas as pd

def load_pickle(filepath):
    """Load a pickle file with compatibility handling"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"   Error loading {filepath}: {e}")
        return None

def calculate_metrics(predictions, groundtruths):
    """Calculate F1, accuracy, and confusion matrix"""
    # Ensure predictions are binary (0 or 1)
    preds_binary = (predictions > 0.5).astype(int) if predictions.max() <= 1.0 else predictions.astype(int)
    gt_binary = groundtruths.astype(int)
    
    f1 = f1_score(gt_binary, preds_binary, average='binary', zero_division=0)
    acc = accuracy_score(gt_binary, preds_binary)
    cm = confusion_matrix(gt_binary, preds_binary)
    
    return {
        'f1': f1,
        'accuracy': acc,
        'confusion_matrix': cm,
        'n_samples': len(groundtruths)
    }

def main():
    base_path = Path("Rahomon_DEAP/Pickles")
    pred_path = base_path / "predictions"
    gt_path = base_path / "groundtruths"
    
    # Get all prediction files
    pred_files = sorted(pred_path.glob("predictions_*.pkl"))
    
    print(f"Found {len(pred_files)} pipeline prediction files\n")
     
    
    results = []
    
    for pred_file in pred_files:
        pipeline_name = pred_file.stem.replace('predictions_', '')
        gt_file = gt_path / f"groundtruths_{pipeline_name}.pkl"
        
        if not gt_file.exists():
            print(f"  Missing groundtruth for {pipeline_name}")
            continue
        
        try:
            # Load predictions and groundtruths
            predictions = load_pickle(pred_file)
            groundtruths = load_pickle(gt_file)
            
            if predictions is None or groundtruths is None:
                print(f"  Could not load data for {pipeline_name}")
                continue
            
            # Check structure
            if isinstance(predictions, dict):
                # Predictions per target (arousal, valence, dominance)
                for target, pred_dict in predictions.items():
                    if target in groundtruths:
                        gt_dict = groundtruths[target]
                        
                        # Combine all subject predictions
                        all_preds = []
                        all_gts = []
                        
                        for subject_id in pred_dict.keys():
                            if subject_id in gt_dict:
                                all_preds.extend(pred_dict[subject_id])
                                all_gts.extend(gt_dict[subject_id])
                        
                        if len(all_preds) > 0:
                            all_preds = np.array(all_preds)
                            all_gts = np.array(all_gts)
                            
                            metrics = calculate_metrics(all_preds, all_gts)
                            
                            results.append({
                                'pipeline': pipeline_name,
                                'target': target,
                                'f1': metrics['f1'],
                                'accuracy': metrics['accuracy'],
                                'n_samples': metrics['n_samples']
                            })
                            
                            print(f" {pipeline_name} | {target:10s} | F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f} | N: {metrics['n_samples']}")
            else:
                # Single array predictions
                metrics = calculate_metrics(predictions, groundtruths)
                results.append({
                    'pipeline': pipeline_name,
                    'target': 'unknown',
                    'f1': metrics['f1'],
                    'accuracy': metrics['accuracy'],
                    'n_samples': metrics['n_samples']
                })
                print(f" {pipeline_name} | F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f} | N: {metrics['n_samples']}")
                
        except Exception as e:
            print(f"  Error processing {pipeline_name}: {str(e)}")
            continue
    
     
    print("\n  SUMMARY STATISTICS")
     
    
    if results:
        df = pd.DataFrame(results)
        
        # Overall statistics
        print(f"\nTotal evaluations: {len(df)}")
        print(f"Mean F1: {df['f1'].mean():.4f} (±{df['f1'].std():.4f})")
        print(f"Median F1: {df['f1'].median():.4f}")
        print(f"Max F1: {df['f1'].max():.4f}")
        print(f"Min F1: {df['f1'].min():.4f}")
        
        # Per-target statistics (if targets exist)
        if 'target' in df.columns and df['target'].iloc[0] != 'unknown':
            print("\n Per-Target Statistics:")
            for target in df['target'].unique():
                target_df = df[df['target'] == target]
                print(f"\n{target.upper()}:")
                print(f"  Mean F1: {target_df['f1'].mean():.4f} (±{target_df['f1'].std():.4f})")
                print(f"  Median F1: {target_df['f1'].median():.4f}")
                print(f"  Max F1: {target_df['f1'].max():.4f}")
                print(f"  Best pipeline: {target_df.loc[target_df['f1'].idxmax(), 'pipeline']}")
        
        # Top 10 pipelines
        print("\n  TOP 10 PIPELINES (by F1):")
        top10 = df.nlargest(10, 'f1')
        for idx, row in top10.iterrows():
            target_str = f"[{row['target']}]" if row['target'] != 'unknown' else ""
            print(f"  {row['f1']:.4f} - {row['pipeline']} {target_str}")
        
        # Save results
        output_file = "Rahomon_DEAP/calculated_metrics_from_pickles.csv"
        df.to_csv(output_file, index=False)
        print(f"\n Results saved to: {output_file}")
        
        # Answer the key question
         
        print(" KEY FINDING: Does this support 90%+ F1 claims?")
         
        
        above_90 = df[df['f1'] >= 0.90]
        above_85 = df[df['f1'] >= 0.85]
        
        print(f"\nPipelines with F1 >= 0.90 (90%): {len(above_90)} out of {len(df)} ({len(above_90)/len(df)*100:.1f}%)")
        print(f"Pipelines with F1 >= 0.85 (85%): {len(above_85)} out of {len(df)} ({len(above_85)/len(df)*100:.1f}%)")
        
        if len(above_90) > 0:
            print(f"\n YES - Found {len(above_90)} evaluations achieving 90%+ F1:")
            for idx, row in above_90.iterrows():
                print(f"   • {row['f1']:.4f} - {row['pipeline']} [{row['target']}]")
        else:
            print(f"\n  NO - Maximum F1 achieved: {df['f1'].max():.4f} ({df['f1'].max()*100:.2f}%)")
            print(f"   This is {(0.90 - df['f1'].max())*100:.1f} percentage points below 90%")
    else:
        print("  No results to analyze")

if __name__ == "__main__":
    main()
