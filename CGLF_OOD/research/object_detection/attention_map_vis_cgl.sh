# Run Script
# bash attention_map_vis_cgl.sh "$PYTHONPATH:/home/saket/Dense/CGLF_OOD/research:/home/saket/Dense/CGLF_OOD/research/slim" "1" "2" "training_all_4c2_2gl_copy" "attention_map_all_4c2_2gl" "legacy"
# 1. Export Python Path
# 2, 3. GPUs
# 4. Training Folder
# 5. Results Folder
# 6. Folder to Edit (Legacy)
#from pathlib import Path
# Message
echo "Starting..."
# Set Python Path
#print(Path.cwd())
export PYTHONPATH="$1"
# Set Conda GPU's
export CUDA_VISIBLE_DEVICES=$2,$3
# Activate Conda
#conda activate tensorflow-gpu
# Message
echo "Set Python Path $PYTHONPATH and Conda GPU's $CUDA_VISIBLE_DEVICES Competed."

# LightFog Weather
rm nohup.out
cd ./$6/
cp evaluator_aw.py evaluator.py
cd ..
rm -r $5
mkdir $5
cd $5
mkdir LightFog
mkdir DenseFog
mkdir Snow
mkdir Clear
cd ..

# Run Evaluation Script for Day Easy 
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/LightFog/global_gated_vis_LF_day_easy.png
mv weights_original_image.png ./$5/LightFog/weights_original_image_LF_day_easy.png
mv weights_gated.png ./$5/LightFog/weights_gated_LF_day_easy.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/LightFog/global_camera_vis_LF_day_easy.png
mv weights_camera.png ./$5/LightFog/weights_camera_LF_day_easy.png
echo "Weights_Camera saved.."
echo "Visualization saved.."

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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/LightFog/global_gated_vis_LF_day_mod.png
mv weights_original_image.png ./$5/LightFog/weights_original_image_LF_day_mod.png
mv weights_gated.png ./$5/LightFog/weights_gated_LF_day_mod.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/LightFog/global_camera_vis_LF_day_mod.png
mv weights_camera.png ./$5/LightFog/weights_camera_LF_day_mod.png
echo "Weights_Camera saved.."
echo "Visualization saved.."

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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/LightFog/global_gated_vis_LF_day_hard.png
mv weights_original_image.png ./$5/LightFog/weights_original_image_LF_day_hard.png
mv weights_gated.png ./$5/LightFog/weights_gated_LF_day_hard.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/LightFog/global_camera_vis_LF_day_hard.png
mv weights_camera.png ./$5/LightFog/weights_camera_LF_day_hard.png
echo "Weights_Camera saved.."
echo "Visualization saved.."

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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/LightFog/global_gated_vis_LF_night_hard.png
mv weights_original_image.png ./$5/LightFog/weights_original_image_LF_night_hard.png
mv weights_gated.png ./$5/LightFog/weights_gated_LF_night_hard.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/LightFog/global_camera_vis_LF_night_hard.png
mv weights_camera.png ./$5/LightFog/weights_camera_LF_night_hard.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/LightFog/global_gated_vis_LF_night_mod.png
mv weights_original_image.png ./$5/LightFog/weights_original_image_LF_night_mod.png
mv weights_gated.png ./$5/LightFog/weights_gated_LF_night_mod.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/LightFog/global_camera_vis_LF_night_mod.png
mv weights_camera.png ./$5/LightFog/weights_camera_LF_night_mod.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
# Running Overlay Script
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/LightFog/global_gated_vis_LF_night_easy.png
mv weights_original_image.png ./$5/LightFog/weights_original_image_LF_night_easy.png
mv weights_gated.png ./$5/LightFog/weights_gated_LF_night_easy.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/LightFog/global_camera_vis_LF_night_easy.png
mv weights_camera.png ./$5/LightFog/weights_camera_LF_night_easy.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py
#import os
#print(Path.cwd())
mv vis.png ./$5/DenseFog/global_gated_vis_DF_night_easy.png
mv weights_original_image.png ./$5/DenseFog/weights_original_image_DF_night_easy.png
mv weights_gated.png ./$5/DenseFog/weights_gated_DF_night_easy.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/DenseFog/global_camera_vis_DF_night_easy.png
mv weights_camera.png ./$5/DenseFog/weights_camera_DF_night_easy.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/DenseFog/global_gated_vis_DF_night_mod.png
mv weights_original_image.png ./$5/DenseFog/weights_original_image_DF_night_mod.png
mv weights_gated.png ./$5/DenseFog/weights_gated_DF_night_mod.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/DenseFog/global_camera_vis_DF_night_mod.png
mv weights_camera.png ./$5/DenseFog/weights_camera_DF_night_mod.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/DenseFog/global_gated_vis_DF_night_hard.png
mv weights_original_image.png ./$5/DenseFog/weights_original_image_DF_night_hard.png
mv weights_gated.png ./$5/DenseFog/weights_gated_DF_night_hard.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/DenseFog/global_camera_vis_DF_night_hard.png
mv weights_camera.png ./$5/DenseFog/weights_camera_DF_night_hard.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/DenseFog/global_gated_vis_DF_day_hard.png
mv weights_original_image.png ./$5/DenseFog/weights_original_image_DF_day_hard.png
mv weights_gated.png ./$5/DenseFog/weights_gated_DF_day_hard.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/DenseFog/global_camera_vis_DF_day_hard.png
mv weights_camera.png ./$5/DenseFog/weights_camera_DF_day_hard.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/DenseFog/global_gated_vis_DF_day_mod.png
mv weights_original_image.png ./$5/DenseFog/weights_original_image_DF_day_mod.png
mv weights_gated.png ./$5/DenseFog/weights_gated_DF_day_mod.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/DenseFog/global_camera_vis_DF_day_mod.png
mv weights_camera.png ./$5/DenseFog/weights_camera_DF_day_mod.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/DenseFog/global_gated_vis_DF_day_easy.png
mv weights_original_image.png ./$5/DenseFog/weights_original_image_DF_day_easy.png
mv weights_gated.png ./$5/DenseFog/weights_gated_DF_day_easy.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/DenseFog/global_camera_vis_DF_day_easy.png
mv weights_camera.png ./$5/DenseFog/weights_camera_DF_day_easy.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/Snow/global_gated_vis_S_night_easy.png
mv weights_original_image.png ./$5/Snow/weights_original_image_S_night_easy.png
mv weights_gated.png ./$5/Snow/weights_gated_S_night_easy.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/Snow/global_camera_vis_S_night_easy.png
mv weights_camera.png ./$5/Snow/weights_camera_S_night_easy.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/Snow/global_gated_vis_S_night_mod.png
mv weights_original_image.png ./$5/Snow/weights_original_image_S_night_mod.png
mv weights_gated.png ./$5/Snow/weights_gated_S_night_mod.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/Snow/global_camera_vis_S_night_mod.png
mv weights_camera.png ./$5/Snow/weights_camera_S_night_mod.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/Snow/global_gated_vis_S_night_hard.png
mv weights_original_image.png ./$5/Snow/weights_original_image_S_night_hard.png
mv weights_gated.png ./$5/Snow/weights_gated_S_night_hard.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/Snow/global_camera_vis_S_night_hard.png
mv weights_camera.png ./$5/Snow/weights_camera_S_night_hard.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/Snow/global_gated_vis_S_day_hard.png
mv weights_original_image.png ./$5/Snow/weights_original_image_S_day_hard.png
mv weights_gated.png ./$5/Snow/weights_gated_S_day_hard.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/Snow/global_camera_vis_S_day_hard.png
mv weights_camera.png ./$5/Snow/weights_camera_S_day_hard.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/Snow/global_gated_vis_S_day_mod.png
mv weights_original_image.png ./$5/Snow/weights_original_image_S_day_mod.png
mv weights_gated.png ./$5/Snow/weights_gated_S_day_mod.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/Snow/global_camera_vis_S_day_mod.png
mv weights_camera.png ./$5/Snow/weights_camera_S_day_mod.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/Snow/global_gated_vis_S_day_easy.png
mv weights_original_image.png ./$5/Snow/weights_original_image_S_day_easy.png
mv weights_gated.png ./$5/Snow/weights_gated_S_day_easy.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/Snow/global_camera_vis_S_day_easy.png
mv weights_camera.png ./$5/Snow/weights_camera_S_day_easy.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/Clear/global_gated_vis_clear_day_easy.png
mv weights_original_image.png ./$5/Clear/weights_original_image_clear_day_easy.png
mv weights_gated.png ./$5/Clear/weights_gated_clear_day_easy.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/Clear/global_camera_vis_clear_day_easy.png
mv weights_camera.png ./$5/Clear/weights_camera_clear_day_easy.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/Clear/global_gated_vis_clear_day_mod.png
mv weights_original_image.png ./$5/Clear/weights_original_image_clear_day_mod.png
mv weights_gated.png ./$5/Clear/weights_gated_clear_day_mod.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/Clear/global_camera_vis_clear_day_mod.png
mv weights_camera.png ./$5/Clear/weights_camera_clear_day_mod.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/Clear/global_gated_vis_clear_day_hard.png
mv weights_original_image.png ./$5/Clear/weights_original_image_clear_day_hard.png
mv weights_gated.png ./$5/Clear/weights_gated_clear_day_hard.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/Clear/global_camera_vis_clear_day_hard.png
mv weights_camera.png ./$5/Clear/weights_camera_clear_day_hard.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/Clear/global_gated_vis_clear_night_hard.png
mv weights_original_image.png ./$5/Clear/weights_original_image_clear_night_hard.png
mv weights_gated.png ./$5/Clear/weights_gated_clear_night_hard.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/Clear/global_camera_vis_clear_night_hard.png
mv weights_camera.png ./$5/Clear/weights_camera_clear_night_hard.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/Clear/global_gated_vis_clear_night_mod.png
mv weights_original_image.png ./$5/Clear/weights_original_image_clear_night_mod.png
mv weights_gated.png ./$5/Clear/weights_gated_clear_night_mod.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/Clear/global_camera_vis_clear_night_mod.png
mv weights_camera.png ./$5/Clear/weights_camera_clear_night_mod.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
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
# Initial File Edit
cd ./$6/
filename="evaluator.py"
search="attention_weights_camera"
replace="attention_weights_gated"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
filename="evaluator.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
# Running Overlay Script
python vis.py 
mv vis.png ./$5/Clear/global_gated_vis_clear_night_easy.png
mv weights_original_image.png ./$5/Clear/weights_original_image_clear_night_easy.png
mv weights_gated.png ./$5/Clear/weights_gated_clear_night_easy.png
echo "Weights Original Image saved.."
echo "Weights Gated saved.."
# File Edit 1
cd ./$6/
filename="evaluator.py"
search="attention_weights_gated"
replace="attention_weights_camera"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# File Edit 2
filename="evaluator.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
cd ..
# File Edit 3
filename="vis.py"
search="weights_gated.png"
replace="weights_camera.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
# Run Eval Script
nohup python eval.py --logtostderr --checkpoint_dir=$4 --eval_dir=eval_auto --pipeline_config_path=$4/faster_rcnn_inception_v2_pets.config &
sleep 50 ; kill %%
python vis.py 
mv vis.png ./$5/Clear/global_camera_vis_clear_night_easy.png
mv weights_camera.png ./$5/Clear/weights_camera_clear_night_easy.png
echo "Weights_Camera saved.."
echo "Visualization saved.."
echo "Completed Evaluation Night 3 (Easy)"
echo "Completed Lightfog Evaluation"
echo "Finished running execution script"

# Final File Edit
cd ./$6/
cp evaluator_c.py evaluator.py
cd ..
filename="vis.py"
search="weights_camera.png"
replace="weights_gated.png"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" -e "s/$search/$replace/" $filename
fi
