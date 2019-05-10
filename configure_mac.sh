PROJECT_DIR=~/Github/shapeology_code
virtualenv="shapeology_venv"
##################################################

red='\e[1;31m'
purple='\e[1;35m'
green='\e[1;32m'
cyan='\e[1;36m'
NC='\033[0m' # No Color

export REPO_DIR=$PROJECT_DIR/scripts/

# FOR UCSD BRAIN
export ROOT_DIR=~/BstemAtlasDataBackup/ucsd_brain/

venv_dir=~/Github/venv/$virtualenv

if [ ! -d $venv_dir ]; then
        echo ""
        echo -e "${green}Creating a virtualenv environment${NC}"
        virtualenv -p python3 $venv_dir
        
        echo ""
        echo -e "${green}Activating the virtualenv environment${NC}"
        source $venv_dir/bin/activate
        
        echo ""
        echo -e "${green}[virtualenv] Installing Python packages${NC}"
        sudo pip3 install opencv-python
        sudo pip3 install astropy
        sudo pip3 install scipy
        sudo pip3 install scikit-image
        sudo pip3 install photutils
        sudo pip3 install glymur
        sudo pip3 install pydiffmap
        sudo pip3 install psutil
        sudo pip3 install pyyaml
        sudo pip3 install scikit-learn==0.20.0
fi

echo ""
echo -e "${green}Activating the virtualenv environment${NC}"
source $venv_dir/bin/activate
 
