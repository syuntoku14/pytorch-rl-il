# umask 0002 is to change the permission to a normal user

run_docker() {
docker run --rm -it --privileged \
	--net=host \
	--ipc=host \
	-v ~/RL_ws:/root/RL_ws \
	-v ~/pytorch-rl-il:/root/pytorch-rl-il \
	-e DISPLAY=:0 \
	--name rl \
	--shm-size 16G \
	--entrypoint "" \
	syuntoku/rl_ws:rlil bash -c "umask 0002 && bash"
}

run_docker_gpu() {
docker run --rm -it --privileged \
	--net=host \
	--ipc=host \
	-v ~/RL_ws:/root/RL_ws \
	-v ~/pytorch-rl-il:/root/pytorch-rl-il \
	-e DISPLAY=:0 \
	--name rl \
	--shm-size 16G \
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
