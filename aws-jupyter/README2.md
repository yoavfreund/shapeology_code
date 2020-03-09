## Install dependencies

Run `pip install -r requirements.txt` to install the required python packages.

## Requirements

These scripts requires Python 3 and the `awscli` package for Python 3. In addition,
the `create-cluster.py` script uses an AMI image that located in the region `us-east-1`.
So please make sure that your local environment is set up to use that region.
If it is not, you can change it by running `aws configure` in your commandline, and type
`us-east-1` for the region when prompted.


### AWS Credential

The scripts in this repository requires a `credentials.yml` file in following format:

```yaml
Arbitrary Name:
  access_key_id: your_aws_access_key_id
  secret_access_key: your_aws_secret_access_key
  key_name: your_ec2_key_pair_name
  ssh_key: /path/to/the/ec2/key/pair/file
```

The credential file in the Spark Notebook project can be directly used here.

The credential file (or a soft link to it) should be located in the same folder where
you invoke these scripts (i.e. you should be able to see it using `ls .` command).
The credential file must always **stay private** and not be shared. Remember to add
`credential.yml` to the `.gitignore` file of your project so that this
file would not be pushed to GitHub.


## Help message

Run any script in this directory with `-h` argument will print the help message of the script.


## Create a new cluster

`create-cluster.py` creates a cluster on the `m3.xlarge` instance using an AMI based on Ubuntu.

### Example
```bash
./create-cluster.py -c 2 --name testing
```


## Check if a cluster is ready

`check-cluster.py` checks if a cluster is up and running. In addition, it also creates a
`neighbors.txt` file which contains the IP addresses of all the instances in the cluster.

### Example
```bash
./check-cluster.py --name testing
```


## Terminate a cluster

`terminate-cluster.py` terminates a cluster by stopping and terminating all instances
in this cluster.

### Example
```bash
./terminate-cluster.py --name testing
```


## Run a script on a cluster

`run-cluster.py` runs a given script on all instances in the cluster.
It starts the script in the background, and redirect the stdout/stderr
into a file on the instances which can be checked later.
Thus it terminates does not necessarily mean the script has finshed executing on the cluster.
In addition, it only launches the script on all instances, but does _not_ check if the script
executes without error.

### Example
```bash
./run-cluster.py --script ./script-examples/hello-world.sh
```


## Send configuration files to all instances in a cluster

In most cases, we would like the different workers/instances in a cluster run with
different parameters. We can achieve that by generating a different configuration file
for each worker, and letting the program read its parameter from this file.
The script `send-configs.py` is used for sending the configuration files to the workers.
Please refer to the Find Prime Number examples in `/example` for the demonstration of using
this script.

### Example
After generating a cluster with `N` workers, one can write a custom script to generate `N`
configuration files, one for each worker, and save all configuration files in the some directory
(e.g. `./example-configs/`). After that, run following command

```bash
./send-configs.py --config ./example-configs/
```


## Retrieve files from all instances in a cluster

`retrieve-files.py` retrieve files from the same location on all instances of a cluster.
It can be used to collect the output of the program from the workers.
A local directory for saving the downloaded files should be provided to this script.
This script will create a separate sub-directory for each worker and download its files
to this sub-directory.

### Example
```bash
./retrieve-files.py --remote /home/ubuntu/workspace/rust-tmsn/output.txt --local ./_result/
```
