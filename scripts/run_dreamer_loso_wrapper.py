import sys
import runpy

# Configure arguments for the LOSO runner
sys.argv = [
    'run_dreamer_loso.py',
    '--data_path', r'C:\Users\rgangolu\OneDrive - Infor\rashomon-eeg-emotions-recognition\data\DREAMER\DREAMER.mat',
    '--preproc', 'wavelet',
    '--epochs', '5',
    '--out', 'dreamer_loso_results.json'
]

# Execute the script as __main__
runpy.run_path('scripts/run_dreamer_loso.py', run_name='__main__')
