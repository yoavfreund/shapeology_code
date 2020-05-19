## Run on multiple AWS instances
### About
This script is to launches a cluster on AWS EC2 instances and 
send scripts to run on them parallelly. It is based on a python module,
[aws-jupyter](https://github.com/arapat/aws-jupyter).

### Requirements
* aws-jupyter
* datajoint

### Structure
The structure is as follows:
* **credentials.yaml**: Specifies the credentials used for creating EC2 instances. 
Format:
```yaml
Arbitrary Name:
  access_key_id: your_aws_access_key_id
  secret_access_key: your_aws_secret_access_key
  key_name: your_ec2_key_pair_name
  ssh_key: /path/to/the/ec2/key/pair/file
```
* **install-project.sh**: Customizes the commands to run on cloud. Example:
```bash
cd ~/data/Github/
## Download the project
git clone https://github.com/yoavfreund/shapeology_code.git
## Export environment variables on cloud
export SHAPOLOGY_DIR=~/data/Github/shapeology_code
export REPO_DIR=$SHAPOLOGY_DIR/scripts/
export ROOT_DIR=~/data/BstemAtlasDataBackup/ucsd_brain/
export VAULT=~/data/VaultBrain/
export venv_dir=~/data/venv/shapeology_venv
## Activate the virtual environment and run the script
source shapeology_code/virtual_env.sh
cd shapeology_code/scripts/Cell_datajoint/
python Cells_extract_datajoint.py 'shape_params.yaml' 'DK39'
```
* **Vault**: Contains all the credential files for EC2 instances. See [details](README.md).
* **Aws-jupyter.py**: Get the process done sequentially. Launches a cluster and check the status of it until all instances are running. 

### Usage
1.Run `aws-jupyter config` to set the configuration file. You need an AMI (a template instance) in advance.
```bash
Region: your_ami_region
Instance type: your_ami_instance_type
AMI:    your_ami_id
Credential: path_to_credentials.yaml
```
2.Create a cluster, transfer **Vault** to instances of the cluster and send `..\install-project.sh` to run.
```bash
python Aws-jupyter.py cluster_name number script_fp
```
```
optional arguments:
  cluster_name   The name of a cluster, type=str, default='test'
  number         The number of instances to create, type=int, default=10
  script_fp      Path to bash file to run on cloud, default=`..\install-project.sh`
```
3.Terminate the cluster.
```bash
aws-jupyter terminate --name cluster_name
```

