sudo apt install python3-pip grass-dev
sudo python3 -m pip install sentinelsat
sudo chown odse:odse /usr/local/grass8.0.dev-x86_64-pc-linux-gnu-03_09_2021/ -R
sudo cp -r /usr/local/grass8.0.dev-x86_64-pc-linux-gnu-03_09_2021/utils/* /usr/local/grass8.0.dev-x86_64-pc-linux-gnu-03_09_2021/dist.x86_64-pc-linux-gnu/utils/
