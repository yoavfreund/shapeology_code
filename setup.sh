sudo apt-get install python3-setuptools
sudo easy_install3 pip
sudo mv /usr/local/bin/pip /usr/local/bin/pip3
 
sudo pip3.5 install --user opencv-contrib-python
sudo pip3.5 install astropy
sudo pip3.5 install scipy
sudo pip3.5 install scikit-image
sudo pip3.5 install photutils
sudo pip3.5 install glymur

sudo pip3.5 install awscli

aws configure (or find a better way)

mkdir ~/data
aws s3 cp s3://mousebrainatlas-rawdata/CSHL_data
/Tmp/MD657-F23-2017.02.17-23.13.17_MD657_2_0068_lossless.jp2  ./data/

