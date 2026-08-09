University Server
(base) lizening@acc28b43e0d2:~$ nvidia-smi
Sun Aug  9 07:02:40 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.163.01             Driver Version: 550.163.01     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A40                     Off |   00000000:A2:00.0 Off |                    0 |
|  0%   76C    P0            232W /  300W |   27020MiB /  46068MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
(base) lizening@acc28b43e0d2:~$ python --version
Python 3.13.13
(base) lizening@acc28b43e0d2:~$ which python
/home/lizening/miniconda3/bin/python
(base) lizening@acc28b43e0d2:~$ git --version
git version 2.34.1
(base) lizening@acc28b43e0d2:~$ df -h
Filesystem                                     Size  Used Avail Use% Mounted on
overlay                                        271G   77G  183G  30% /
tmpfs                                           64M     0   64M   0% /dev
tmpfs                                           63G     0   63G   0% /sys/fs/cgroup
shm                                             64M     0   64M   0% /dev/shm
/dev/mapper/3600a098038314a79383f574f486e6270   98G   14G   80G  15% /home
/dev/mapper/ubuntu--vg-ubuntu--lv              271G   77G  183G  30% /etc/hosts
tmpfs                                           63G   12K   63G   1% /proc/driver/nvidia
tmpfs                                           13G  4.8M   13G   1% /run/nvidia-persistenced/socket
tmpfs                                           63G     0   63G   0% /proc/acpi
tmpfs                                           63G     0   63G   0% /proc/scsi
tmpfs                                           63G     0   63G   0% /sys/firmware
tmpfs                                           63G     0   63G   0% /sys/devices/virtual/powercap
(base) lizening@acc28b43e0d2:~$ echo $HOME
/home/lizening

(base) lizening@acc28b43e0d2:~$ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
name, memory.total [MiB], driver_version
NVIDIA A40, 46068 MiB, 550.163.01

(base) lizening@acc28b43e0d2:~$ conda --version
conda 26.3.2