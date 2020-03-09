This script launches a cluster on AWS EC2 instances and starts a Jupyter notebook on them.

## Install

Run `pip install -r requirements.txt` to install the required python packages.

These scripts requires Python 3 and the `awscli` package for Python 3.


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


## Usage 

Run `./aws-jupyter.py -h` will print the help message of the script.


1. Run configuration - run `./aws-jupyter.py config` to set up the script.
2. Create a cluster - run `./aws-jupyter.py create -c <N> --name <NAME>` to create a cluster
named <NAME> with <N> nodes. Later the cluster name is required to check the cluster status and
to terminate the cluster.
If on-demand instances are needed, please append `--ondemand` option.
3. Check a cluster - run `./aws-jupyter.py check --name <NAME>` to check the status of a cluster.
If the cluster is ready, you will see the URL to the Jupyter notebook.
4. Terminate a cluster - run `./aws-jupyter.py terminate --name <NAME>` to terminate a cluster.

