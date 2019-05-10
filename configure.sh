PROJECT_DIR=/data/Github/shapeology_code
virtualenv="shapeology_venv"
##################################################

red='\e[1;31m'
purple='\e[1;35m'
green='\e[1;32m'
cyan='\e[1;36m'
NC='\033[0m' # No Color

export REPO_DIR=$PROJECT_DIR/scripts/

# FOR UCSD BRAIN
export ROOT_DIR=/data/BstemAtlasDataBackup/ucsd_brain/

venv_dir=/data/venv/$virtualenv

if [ ! -d $venv_dir ]; then
        echo ""
        echo -e "${green}Creating a virtualenv environment${NC}"
        virtualenv -p python3 $venv_dir
        echo ""
	echo -e "${green}Activating the virtualenv environment${NC}"
	source $venv_dir/bin/activate
	echo ""
        echo -e "${green}[virtualenv] Installing Python packages${NC}"
        pip3 install opencv-python
        pip3 install astropy
        pip3 install scipy
        pip3 install scikit-image
        pip3 install photutils
        pip3 install glymur
fi

echo ""
echo -e "${green}Activating the virtualenv environment${NC}"
source $venv_dir/bin/activate
 
