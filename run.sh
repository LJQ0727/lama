export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
python mask_extract/maskExtract.py
python asset_manager.py
python bin/predict.py
