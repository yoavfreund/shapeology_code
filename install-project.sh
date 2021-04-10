cd ~/data/Github/
#sleep $(shuf -i 1-100 -n 1)
rm -rf shapeology_code
git clone -b kui_dev --single-branch https://github.com/yoavfreund/shapeology_code.git

export SHAPEOLOGY_DIR=~/data/Github/shapeology_code
export ROOT_DIR=~/data/BstemAtlasDataBackup/ucsd_brain/
export VAULT=~/data/VaultBrain/
export venv=~/data/venv/

source shapeology_code/virtual_env.sh
pip install awscli
pip install shapely
pip install mxnet
cd shapeology_code/scripts/Cell_datajoint/

#python Cells_extract_datajoint.py 'shape_params.yaml' 'DK39'
python Shift_score_datajoint.py 'DK52'
