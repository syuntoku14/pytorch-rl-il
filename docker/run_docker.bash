# umask 0002 is to change the permission to a normal user

run_docker() {
docker run --rm -it \
	-p 6080:6080 \
	-p 8888:8888 \
	-p 6006:6006 \
	-p 5678:5678 \
	-p 8265:8265 \
	-v ~/RL_ws:/root/RL_ws \
	-v ~/pytorch-rl-il:/root/pytorch-rl-il \
	-e DISPLAY=:0 \
	--name rl \
	--shm-size 256G \
	--entrypoint "" \
	syuntoku/rl_ws:rlil bash -c "umask 0002 && bash"
}

run_docker_gpu() {
docker run --rm -it \
	-p 6080:6080 \
	-p 8888:8888 \
	-p 6006:6006 \
	-p 5678:5678 \
	-p 8265:8265 \
	-v ~/RL_ws:/root/RL_ws \
	-v ~/pytorch-rl-il:/root/pytorch-rl-il \
	-e DISPLAY=:0 \
	--name rl \
	--shm-size 256G \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	--gpus=all \
	--entrypoint "" \
	syuntoku/rl_ws:rlil bash -c "umask 0002 && bash"
}

getopts "n" OPT
case $OPT in
	n ) echo "--runtime=nvidia"
		run_docker_gpu ;;
	? )	echo "Without gpu"
		run_docker ;;
esac
