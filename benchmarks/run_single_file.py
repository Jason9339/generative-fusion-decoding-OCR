import numpy as np
import argparse
import json

from gfd.gfd import Breezper
from gfd.utils import process_config, combine_config

def parse_args():
    parser = argparse.ArgumentParser(description="Override config settings with command-line arguments.")
    parser.add_argument('--model_name', type=str, help='The model for testing the benchmark dataset')
    parser.add_argument('--setting', type=str, help='benchmark dataset settings for specified model')
    parser.add_argument('--image_file_path', type=str, help='Path to the image file sample')
    parser.add_argument('--result_output_path', type=str, help='Path to save dataset with predictions from the model')
    return parser.parse_args()

def main():
    setting_configs = {
        'gfd': {
            'ocr-en': process_config('config_files/model/gfd-ocr-en.yaml')
        }
    }
    args = parse_args()
    setting_config = setting_configs[args.model_name][args.setting]
    prompt_config = process_config('config_files/prompt/ocr-default-prompt.yaml')
    combined_config = combine_config(prompt_config, setting_config)

    model = Breezper(combined_config)
    result = model.get_transcription(
        args.image_file_path,
        ocr_prompt=combined_config.ocr_prompt,
        llm_prompt=combined_config.llm_prompt
    )
    print(f'Result: {result}')

    with open(args.result_output_path, 'w') as f:
        json.dump(result, f, ensure_ascii=False)

if __name__== '__main__':
    main()    
