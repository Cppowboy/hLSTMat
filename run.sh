now=$(date "+%Y%m%d_%H%M%S")

# export ...
#export PATH=/mnt/lustre/panyinxu/anaconda2/bin:$PATH
#export LD_LIBRARY_PATH=/mnt/lustre/panyinxu/.local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/mnt/lustre/panyinxu/anaconda/bin:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/panyinxu/.local/lib:/mnt/lustre/share/cuda-8.0-cudnn5.1:$LD_LIBRARY_PATH
export CPATH=$CPATH:/mnt/lustre/panyinxu/.local/include
export LIBRARY_PATH=$LIBRARY_PATH:/mnt/lustre/panyinxu/.local/lib

if [ ! -d "log" ];then
	mkdir log
fi

jobname=hLSTMat
num_gpus=4
THEANO_FLAGS="device=gpu0,optimizer_including=cudnn,floatX=float32,cuda.root=/mnt/lustre/panyinxu/cuda-8.0,blas.ldflasg=,nvcc.flags=-D_FORCE_INLINES"

srun -p TITANXP --job-name=$jobname --gres=gpu:$num_gpus --kill-on-bad-exit=1 \
python train_model.py \
2>&1| tee log/$jobname-$now.log &\
