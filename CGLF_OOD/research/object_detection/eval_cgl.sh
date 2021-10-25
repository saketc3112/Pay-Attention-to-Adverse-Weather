# Run Script
# bash eval.sh "$PYTHONPATH:/home/saket/Dense/CGF_OOD/research:/home/saket/Dense/CGF_OOD/research/slim" "6" "7" "training_all_4c4_2gl" "results_all_4c4_2gl"
# 1. Export Python Path
# 2, 3. GPUs
# 4. Training Folder
# 5. Results Folder

# Message
echo "Starting..."
# Set Python Path
export PYTHONPATH="$1"
# Set Conda GPU's
export CUDA_VISIBLE_DEVICES=$2,$3
# Activate Conda
#conda activate tensorflow-gpu
# Message
echo "Set Python Path $PYTHONPATH and Conda GPU's $CUDA_VISIBLE_DEVICES Competed."
# Checkpoints
#echo $PYTHONPATH
#echo $CUDA_VISIBLE_DEVICES
#echo "Current Path is ${pwd}"

# LightFog Weather
rm nohup.out
rm -r $5
mkdir $5
# Run Evaluation Script for Day Easy 
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
#mkdir $5
mv nohup.out ./$5/record_LF_day_easy.txt
echo "Completed Evaluation Day 1 (Easy)"

# Open Eval Config File
# Assign filename
#cd ./training_all_4c4_2gl/
cd ./$4/
filename="faster_rcnn_inception_v2_pets.config"
# Take the search string
#read -p "Enter the search string: " search
search="test_lightfog_camera_gated_lidar_day_4c_dbe"
# Take the replace string
#read -p "Enter the replace string: " replace
replace="test_lightfog_camera_gated_lidar_day_4c_dbm"

if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi

# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Day Moderate
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_LF_day_mod.txt
echo "Completed Evaluation Day 2 (Moderate)"

# Open Eval Config File
# Assign filename
#cd ./training_all_4c4_2gl/
cd ./$4/
filename="faster_rcnn_inception_v2_pets.config"

# Take the search string
#read -p "Enter the search string: " search
search="test_lightfog_camera_gated_lidar_day_4c_dbm"

# Take the replace string
#read -p "Enter the replace string: " replace
replace="test_lightfog_camera_gated_lidar_day_4c_dbh"

if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi

# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Day Hard
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_LF_day_hard.txt
echo "Completed Evaluation Day 3 (Hard)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_lightfog_camera_gated_lidar_day_4c_dbh"
replace="test_lightfog_camera_gated_lidar_night_4c_dbh"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Night Hard 
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_LF_night_hard.txt
echo "Completed Evaluation Night 1 (Hard)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_lightfog_camera_gated_lidar_night_4c_dbh"
replace="test_lightfog_camera_gated_lidar_night_4c_dbm"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Night Moderate
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_LF_night_mod.txt
echo "Completed Evaluation Night 2 (Moderate)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_lightfog_camera_gated_lidar_night_4c_dbm"
replace="test_lightfog_camera_gated_lidar_night_4c_dbe"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Night Easy
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_LF_night_easy.txt
echo "Completed Evaluation Night 3 (Easy)"
echo "Completed Lightfog Evaluation"

#
# DenseFog Weather
#

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_lightfog_camera_gated_lidar_night_4c_dbe"
replace="test_densefog_camera_gated_lidar_night_4c_dbe"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Night Easy 
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_DF_night_easy.txt
echo "Completed Evaluation Night 1 (Easy)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_densefog_camera_gated_lidar_night_4c_dbe"
replace="test_densefog_camera_gated_lidar_night_4c_dbm"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Moderate
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_DF_night_mod.txt
echo "Completed Evaluation Night 2 (Moderate)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_densefog_camera_gated_lidar_night_4c_dbm"
replace="test_densefog_camera_gated_lidar_night_4c_dbh"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Night Hard
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_DF_night_hard.txt
echo "Completed Evaluation Night 3 (Hard)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_densefog_camera_gated_lidar_night_4c_dbh"
replace="test_densefog_camera_gated_lidar_day_4c_dbh"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Day Hard 
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_DF_day_hard.txt
echo "Completed Evaluation Day 1 (Hard)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_densefog_camera_gated_lidar_day_4c_dbh"
replace="test_densefog_camera_gated_lidar_day_4c_dbm"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Day Moderate
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_DF_day_mod.txt
echo "Completed Evaluation Day 2 (Moderate)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_densefog_camera_gated_lidar_day_4c_dbm"
replace="test_densefog_camera_gated_lidar_day_4c_dbe"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Day Easy
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_DF_day_easy.txt
echo "Completed Evaluation Day 3 (Easy)"
echo "Completed Densefog Evaluation"

#
# Snow Weather
#

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_densefog_camera_gated_lidar_day_4c_dbe"
replace="test_snow_camera_gated_lidar_night_4c_dbe"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Night Easy 
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_s_night_easy.txt
echo "Completed Evaluation Night 1 (Easy)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_snow_camera_gated_lidar_night_4c_dbe"
replace="test_snow_camera_gated_lidar_night_4c_dbm"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Moderate
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_s_night_mod.txt
echo "Completed Evaluation Night 2 (Moderate)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_snow_camera_gated_lidar_night_4c_dbm"
replace="test_snow_camera_gated_lidar_night_4c_dbh"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Night Hard
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_s_night_hard.txt
echo "Completed Evaluation Night 3 (Hard)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_snow_camera_gated_lidar_night_4c_dbh"
replace="test_snow_camera_gated_lidar_day_4c_dbh"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Day Hard 
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_s_day_hard.txt
echo "Completed Evaluation Day 1 (Hard)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_snow_camera_gated_lidar_day_4c_dbh"
replace="test_snow_camera_gated_lidar_day_4c_dbm"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Day Moderate
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_s_day_mod.txt
echo "Completed Evaluation Day 2 (Moderate)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_snow_camera_gated_lidar_day_4c_dbm"
replace="test_snow_camera_gated_lidar_day_4c_dbe"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Day Easy
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_s_day_easy.txt
echo "Completed Evaluation Day 3 (Easy)"
echo "Completed Snow Evaluation"

#
# Clear Weather
#

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_snow_camera_gated_lidar_day_4c_dbe"
replace="test_clear_camera_gated_lidar_day_4c_dbe"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Day Easy 
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_clear_day_easy.txt
echo "Completed Evaluation Day 1 (Easy)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_clear_camera_gated_lidar_day_4c_dbe"
replace="test_clear_camera_gated_lidar_day_4c_dbm"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Moderate
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_clear_day_mod.txt
echo "Completed Evaluation Day 2 (Moderate)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_clear_camera_gated_lidar_day_4c_dbm"
replace="test_clear_camera_gated_lidar_day_4c_dbh"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Day Hard
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_clear_day_hard.txt
echo "Completed Evaluation Day 3 (Hard)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_clear_camera_gated_lidar_day_4c_dbh"
replace="test_clear_camera_gated_lidar_night_4c_dbh"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Night Hard 
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_clear_night_hard.txt
echo "Completed Evaluation Day 1 (Hard)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_clear_camera_gated_lidar_night_4c_dbh"
replace="test_clear_camera_gated_lidar_night_4c_dbm"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Night Moderate
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_clear_night_mod.txt
echo "Completed Evaluation Night 2 (Moderate)"

cd ./$4/ # File Editing
filename="faster_rcnn_inception_v2_pets.config"
search="test_clear_camera_gated_lidar_night_4c_dbm"
replace="test_clear_camera_gated_lidar_night_4c_dbe"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Back to Root Directory
cd ..
# Message
echo "File Edit Completed"

# Run Evaluation Script for Night (Easy) 
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
mv nohup.out ./$5/record_clear_night_easy.txt
echo "Completed Evaluation Night 3 (Easy)"
echo "Completed Lightfog Evaluation"
echo "Finished running execution script"