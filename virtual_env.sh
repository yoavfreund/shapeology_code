virtualenv="shapeology_venv"
export REPO_DIR=$SHAPEOLOGY_DIR/scripts/        # Subdirectory of SHAPEOLOGY_DIR containing python scripts
##################################################

red='\e[1;31m'
purple='\e[1;35m'
green='\e[1;32m'
cyan='\e[1;36m'
NC='\033[0m' # No Color

venv_dir=$venv/$virtualenv

if [ ! -d $venv_dir ]; then
    echo ""
    echo -e "${green}Creating a virtualenv environment${NC}"
    virtualenv -p python3 $venv_dir

    echo ""
    echo -e "${green}Activating the virtualenv environment${NC}"
    source $venv_dir/bin/activate

    echo ""
    echo -e "${green}[virtualenv] Installing Python packages${NC}"
    pip3 install -r $SHAPEOLOGY_DIR/requirements.txt
        
fi

echo ""
echo -e "${green}Activating the virtualenv environment${NC}"
source $venv_dir/bin/activate
 
