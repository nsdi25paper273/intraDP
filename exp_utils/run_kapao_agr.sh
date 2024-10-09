envs=(indoors outdoors)
offload_methods=(mixed2 all fix flex local)
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
        echo "Running $env $method kapao cases..."
        if [ $method == "local" ]
        then
            bash start_work.sh $method "rosrun kapao pose_follower.py" $env robot2_torch13 $dur kapao False

            bash start_work.sh $method "rosrun agrnav inference_ros.py" $env robot2_torch13 $dur agrnav False
        else
            bash start_work.sh $method "rosrun kapao pose_follower.py" $env robot2_torch13 $dur kapao True  $user $ip $port
            
            bash start_work.sh $method "rosrun agrnav inference_ros.py" $env robot2_torch13 $dur agrnav True  $user $ip $port
        fi 

    done
done




