sudo apt-get install python3-setuptools
sudo /home/ubuntu/.local/bin/easy_install pip
#sudo easy_install3 pip
sudo mv /usr/local/bin/pip /usr/local/bin/pip3
 
sudo pip3.5 install --user opencv-contrib-python
sudo pip3.5 install astropy
sudo pip3.5 install scipy
sudo pip3.5 install scikit-image
sudo pip3.5 install photutils
sudo pip3.5 install glymur

sudo pip3.5 install awscli

aws configure (or find a better way)

sudo chmod 0777 /dev/shm/
cd /dev/shm/
mkdir data
cd data
aws s3 cp  s3://mousebraindata-open/MD657/MD657-F23-2017.02.17-23.13.17_MD657_2_0068_lossless.jp2 .

kdu_expand -i MD657-F23-2017.02.17-23.13.17_MD657_2_0068_lossless.jp2 -o MD657-F23-2017.02.17-23.13.17_MD657_2_0068_lossless.tiff

mkdir tiles
convert MD657-F23-2017.02.17-23.13.17_MD657_2_0068_lossless.tiff -crop 1000x1000  +repage  +adjoin  tiles/tiles_%02d.tif


export LD_LIBRARY_PATH="/home/ubuntu/KDU7A2_Demo_Apps_for_Ubuntu-x86-64_170827":$LD_LIBRARY_PATH;
kdu_expand -i MD657-F23-2017.02.17-23.13.17_MD657_2_0068_lossless.jp2 -o MD657-F23-2017.02.17-23.13.17_MD657_2_0068_lossless.tiff
