# Installation
This documentation describes how to set up Shapeology on your own computer.
## Step 1
Clone current github repository to your own computer.
Final version:
```bash
git clone https://github.com/yoavfreund/shapeology_code.git
```
Current version:
```bash
git clone -b kui_dev --single-branch https://github.com/yoavfreund/shapeology_code.git
```

## Step 2
### Configuration Files
* **environ.sh**: Exports the environment variables used for this project. These variables point to the directories of this deployment. Change them to your paths.
Details:
```bash
export SHAPEOLOGY_DIR=~/Github/shapeology_code/  # Directory where you git clone this project
export ROOT_DIR=~/BstemAtlasDataBackup/ucsd_brain/  # Directory where you save input images and output results
export VAULT=~/Github/VaultBrain/             # Optional, Directory for credential files of AWS and datajoint
export venv=~/Github/venv/              # Directory to create virtual environment
```
Note: It is recommended to create new directories or files out of the project directory. Otherwise may have trouble updating the project.
* **VaultBrain** (optional, for AWS use): A directory created to contain all the credential/deployment specific files for AWS and datajoint.
    * **dj_local_conf.json**: Specifies the credentials and the schema that is to be used for Datajoint connection. Format:
    ```json
    {"database.host": your_host_address,
    "database.password": your_datajoint_password,
    "database.user": your_user_name,
    "database.port": your_port,
    "schema": your_schema_name
    }
    ```
    * **s3-creds.json**: Specifies the credentials for AWS connection. Format:
    ```json
    {"access_key": your_aws_access_key, 
    "secret_key": your_aws_secret_access_key}
    ```
    * **credFiles.yaml**: points to the above two json files. Format:
    ```yaml
    aws_fp: s3-creds.json
    dj_fp: dj_local_conf.json
    ```
* When configuration files are set, run the following commands to activate the environment variables and the virtual environment. 
Please ensure you have installed `virtualenv`.
```bash
source environ.sh
source virtual_env.sh
```
Note: Please check whether you have installed the virtual environment successfully. Clear `$venv/shapeology_venv` before you try again if failed.

## Step 3
### Test
To be continued.
