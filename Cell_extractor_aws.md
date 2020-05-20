# Cell Extractor on Cloud
## About
This documentation describes how to use multiple AWS EC2 instances to speed up the cell extraction step.

The script is to launches multiple AWS EC2 instances, send credential files to the remote instances,
and send scripts to run on them parallelly. This is realized via [aws-jupyter](https://github.com/arapat/aws-jupyter), a python module.

## Requirements
* aws-jupyter
* datajoint

## Installation
**1**. Install `aws-jupyter`
```bash
pip install aws-jupyter
```
**2**. Create `credentials.yaml` which specifies the credentials used for creating EC2 instances. 
Format:
```yaml
Arbitrary Name:
  access_key_id: your_aws_access_key_id
  secret_access_key: your_aws_secret_access_key
  key_name: your_ec2_key_pair_name
  ssh_key: /path/to/the/ec2/key/pair/file
```
**3**. Run `aws-jupyter config` to set the configuration file for aws-jupyter. 
```bash
Region: us-west-1
Instance type: r5.xlarge
AMI:   ami-052aabb224c68baf8
Credential: path_to_credentials.yaml
```
**4**. **VaultBrain**: Contains all the credential files for AWS and datajoint populate. See [Installation.md](Installation.md).

## Structure
The structure is as follows:
* **install-project.sh**: Customizes the commands to run on cloud. Example:
```bash
cd ~/data/Github/
## Download the project
git clone https://github.com/yoavfreund/shapeology_code.git
## Export environment variables on cloud
export SHAPEOLOGY_DIR=~/data/Github/shapeology_code
export ROOT_DIR=~/data/BstemAtlasDataBackup/ucsd_brain/
export VAULT=~/data/VaultBrain/
export venv=~/data/venv/
## Activate the virtual environment and run the script
source shapeology_code/virtual_env.sh
cd shapeology_code/scripts/Cell_datajoint/
python Cells_extract_datajoint.py 'shape_params.yaml' 'DK39'
```

* **scripts/Aws-jupyter.py**: Uses aws-jupyter to launches a cluster, send files and check the status of it until all instances are running. 
See details in `Usage`.

### Usage
**1**. Create a datajoint table.

**2**. Create a cluster, transfer **VaultBrain** to instances of the cluster and send `install-project.sh` to run. 
```bash
python $SHAPEOLOGY_DIR/scripts/Aws-jupyter.py
```
```
optional arguments:
  --name        The name of a cluster, type=str, default='test'
  --number      The number of instances to create, type=int, default=10
  --script      Path to bash file to run on cloud, default=`../install-project.sh`
```
**3**. Terminate the cluster.
```bash
aws-jupyter terminate --name test
```

