print('Data Preparation Examples Ready!')
print('Importing data preparation components...')
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
print('Testing TextPreprocessor...')
from data.text_preprocessor import TextPreprocessor
preprocessor = TextPreprocessor()
sample_text = 'የሰላም እለት ነው'
processed = preprocessor.preprocess(sample_text, 'amharic')
print(f'Original: {sample_text}')
print(f'Processed: {processed}')
print('Checkpoint 6: Data Preparation COMPLETE!')
