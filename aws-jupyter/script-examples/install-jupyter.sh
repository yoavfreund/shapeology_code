# Mount SSD
DISK="/dev/$(lsblk | grep -Eo '^nvme[^ ]+' | head -1)"
if [ "$(blkid -o value -s TYPE $DISK)" != "ext4" ]; then
    yes | sudo mkfs.ext4 $DISK
fi
sudo mkdir -p /ssd
sudo mount $DISK /ssd
sudo chown -R ubuntu /ssd

# sudo apt-get update
# sudo apt-get install -y python3-pip
pip3 list | grep -F jupyter-contrib-nbextensions
if [ $? -eq 1 ]; then
    sudo pip3 install --quiet jupyter_contrib_nbextensions RISE
    /home/ubuntu/.local/bin/jupyter contrib nbextension install --user 2> /dev/null 
    sudo $(which jupyter-nbextension) install rise --py --sys-prefix 2> /dev/null
fi
killall jupyter-notebook
# /home/ubuntu/.local/bin/jupyter notebook --generate-config
# echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/ubuntu/.jupyter/jupyter_notebook_config.py
sleep 1
nohup /home/ubuntu/.local/bin/jupyter notebook --no-browser --port=8888 < /dev/null > /dev/null 2>&1 &
URL=$(dig +short myip.opendns.com @resolver1.opendns.com)
sleep 2

echo
echo "The Jupyter Notebook is running on the cluster at the address below."
echo
echo "Open the following address using the browser on your computer"
echo
echo "  http"$(/home/ubuntu/.local/bin/jupyter notebook list | grep -Po '(?<=http).*(?=::)' | sed "s/\/.*:/\/\/$URL:/")
echo
echo "(If the URL didn't show up, please wait a few seconds and try again.)"
echo