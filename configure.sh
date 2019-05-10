PROJECT_DIR=~/MouseBrainAtlas_dev
virtualenv="shapeology_venv"
##################################################

red='\e[1;31m'
purple='\e[1;35m'
green='\e[1;32m'
cyan='\e[1;36m'
NC='\033[0m' # No Color

export REPO_DIR=$PROJECT_DIR/shapeology/

# FOR UCSD BRAIN
export ROOT_DIR=~/BstemAtlasDataBackup/ucsd_brain/


if [ ! -d $virtualenv ]; then
        echo ""
        echo -e "${green}Creating a virtualenv environment${NC}"
        virtualenv -p python3 $REPO_DIR/$virtualenv
        echo ""
        echo -e "${green}[virtualenv] Installing Python packages${NC}"
        sudo pip3 install --user opencv-contrib-python
        sudo pip3 install astropy
        sudo pip3 install scipy
        sudo pip3 install scikit-image
        sudo pip3 install photutils
        sudo pip3 install glymur
fi

echo ""
echo -e "${green}Activating the virtualenv environment${NC}"
source $REPO_DIR/$virtualenv/bin/activate
 
