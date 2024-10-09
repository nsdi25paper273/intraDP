envs=(indoors outdoors)
offload_methods=(mixed2)
# offload_methods=(local all fix flex mixed2)
# datasets=(CIFAR10 CIFAR10 OxfordIIITPet OxfordIIITPet OxfordIIITPet OxfordIIITPet)
# tasks=(classification classification segmentation segmentation detection detection)
models=(VGG19_BN DenseNet121 ConvNeXt_Large ConvNeXt_Base RegNet_X_16GF)
dur=600
user=user
ip=192.168.50.11
port=12345
for method in ${offload_methods[*]}; do
    for env in ${envs[*]}; do

        if [ $method == "local" ] && [ $env == "outdoors" ]
        then
            echo ""
            echo "Skipping $env $method cases..."
            continue
        fi

        echo ""
        echo "Running $env $method torchvision cases..."
        # if [ $method == "all "]     # only need to add all for torchvision
        # then
        for i in {0..4}; do  # TODO run all
            if [ $method == "local" ]
            then
                bash start_work.sh $method "python3 \$work/ros_ws/src/torchvision/scripts/run_torchvision.py -a ${models[i]} -d ImageNet" $env robot2_torch13 $dur ${models[i]} False
            else
                bash start_work.sh $method "python3 \$work/ros_ws/src/torchvision/scripts/run_torchvision.py -a ${models[i]} -d ImageNet" $env robot2_torch13 $dur ${models[i]} True $user $ip $port
            fi
        done
        # fi

        echo ""
        echo "Running $env $method kapao cases..."
        if [ $method == "local" ]
        then
            bash start_work.sh $method "rosrun kapao pose_follower.py" $env robot2_torch13 $dur kapao False

            bash start_work.sh $method "rosrun agrnav inference_ros.py" $env robot2_torch13 $dur agrnav False
        else
            bash start_work.sh $method "rosrun kapao pose_follower.py" $env robot2_torch13 $dur kapao True  $user $ip $port
            
        #     bash start_work.sh $method "rosrun agrnav inference_ros.py" $env robot2_torch13 $dur agrnav True  $user $ip $port
        fi 

    done
done




