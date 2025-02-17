export PROJECT_HOME=$(pwd)
export PROJECT_NAME=$(basename $PROJECT_HOME)
export ENV_NAME=$(echo $PROJECT_NAME | tr '[:upper:]' '[:lower:]')
export ENV_PATH=./.env/$ENV_NAME

# =============== Color Print ===============
DEFAULT=$(echo -en '\033[0m')
RED=$(echo -en '\033[00;31m')
GREEN=$(echo -en '\033[00;32m')
YELLOW=$(echo -en '\033[00;33m')
CYAN=$(echo -en '\033[00;36m')
BLUE=$(echo -en '\033[00;34m')

function print_base     { echo -e "$1- [$2] $3${DEFAULT}"; }
function print_info     { print_base "$CYAN"    "INFO"      "$1"; }
function print_tip      { print_base "$YELLOW"  "TIP"       "$1"; }
function print_success  { print_base "$GREEN"   "SUCCESS"   "$1"; }
function print_warning  { print_base "$YELLOW"  "WARNING"   "$1"; }
function print_error    { print_base "$RED"     "ERROR"     "$1"; }
# ==========================================


export ENV_PATH=$ENV_PATH.venv

if [ ! -d $ENV_PATH ]; then
    python3 -m venv $ENV_PATH
    print_success "Create Python environment in '$ENV_PATH'"
else
    print_info "Python environment '$ENV_PATH' already exists."
fi

source $ENV_PATH/bin/activate
echo "Run the following command to activate the environment:"
echo "source $ENV_PATH/bin/activate"
echo

python3 -m pip install --upgrade pip

pip install -r requirements.txt