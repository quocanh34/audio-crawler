## Status: not completed ##

- Demo 50-hour dataset at: https://huggingface.co/datasets/quocanh34/youtube_dataset_new4_final

- Running on GPU clouds (vast.ai): 
    - chmod +x run_command.sh
    - ./run_command.sh
    - python main.py

- Running on local:
    - pip install -r requirements.txt 
    - python main.py
    
- Check hours:
    - python utils/calculate_hours.py --dataset_path quocanh34/youtube_dataset_new4_final --wer 0
