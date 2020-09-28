#PBS -l nodes=1:ppn=1:gpus=4:exclusive_process
#PBS -q tiny
#PBS -l walltime=0:20:00
#PBS -N train_model
#PBS -k oe
#PBS -m bea

module load cs/keras/2.1.0-tensorflow-1.4-python-3.5
module load lib/hdf5/1.8.16-gnu-4.9

pip install keras==2.2.4

# apparently we need to keep this line to go to the folder first
cd Emotion_recognition/src/models/

python train_model.py -m resnet50 -d imagenet -c c -r 01
#
## Load new singularity version
#module load devel/singularity/3.0.1
#
## Copy container to your workspace or use the one in my workspace
#cd /beegfs/work/iioba01/keras
#
## Open a shell in the container (mind the --nv switch to include GPU driver etc)
#singularity shell --nv keras-2.2.4.sif
#
## Open python3 shell in the container and try keras
#Singularity keras-2.2.4.sif:/beegfs/work/iioba01/keras> /beegfs/work/knvsm01/Emotion_recognition/src/models/train_model.py -m resnet50 -d imagenet -c c -r 01
