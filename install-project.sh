cd ~/data/Github/
sleep $(shuf -i 1-100 -n 1)
rm -rf shapeology_code
git clone -b kui_dev --single-branch https://github.com/yoavfreund/shapeology_code.git

export SHAPOLOGY_DIR=~/data/Github/shapeology_code/shapeology_code
export REPO_DIR=$SHAPOLOGY_DIR/scripts/
export ROOT_DIR=~/data/BstemAtlasDataBackup/ucsd_brain/
export VAULT=~/data/VaultBrain/
export venv_dir=~/data/venv/shapeology_venv

source shapeology_code/virtual_env.sh
pip install awscli
pip install shapely
pip install mxnet
cd shapeology_code/scripts/Cell_datajoint/

#python HSV_datajoint_v2.py 'AWS' 'MD589'
python Cells_extract_datajoint.py 'shape_params.yaml' 'DK39'
#python Patch_extractor_datajoint.py 'AWS' 'MD585'
# python Sample_features_datajoint.py 'AWS' 'MD585'
#python Thresholds_datajoint.py 'AWS' 'MD589'
#python Sqlite_datajoint.py 'AWS' 'MD589'
# python Scoremap_datajoint.py 'AWS' 'MD594'
# python Cell_mark_datajoint.py 'AWS' 'MD594'
