#echo "Done"
#python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_lightfog_day.txt -s test_lightfog_day_camera_gated_lidar_1c_dbe
#python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_lightfog_night.txt -s test_lightfog_night_camera_gated_lidar_1c_dbe
#python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_densefog_day.txt -s test_densefog_day_camera_gated_lidar_1c_dbe
#python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_densefog_night.txt -s test_densefog_night_camera_gated_lidar_1c_dbe
#python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_snow_day.txt -s test_snow_day_camera_gated_lidar_1c_dbe
#python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_snow_night.txt -s test_snow_night_camera_gated_lidar_1c_dbe
#python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_clear_day.txt -s test_clear_day_camera_gated_lidar_1c_dbe
#python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_clear_night.txt -s test_clear_night_camera_gated_lidar_1c_dbe
cd ./generic_tf_tools # File Editing
filename="data2example_1c_db.py"
search="'easy': 1"
replace="'easy': 0"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" $filename
search="'moderate': 0"
replace="'moderate': 1"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" $filename
fi
cd ..
# Message
echo "File Edit Completed"

#rm nohup.out
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_lightfog_day.txt -s test_lightfog_day_camera_gated_lidar_1c_dbm
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_lightfog_night.txt -s test_lightfog_night_camera_gated_lidar_1c_dbm
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_densefog_day.txt -s test_densefog_day_camera_gated_lidar_1c_dbm
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_densefog_night.txt -s test_densefog_night_camera_gated_lidar_1c_dbm
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_snow_day.txt -s test_snow_day_camera_gated_lidar_1c_dbm
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_snow_night.txt -s test_snow_night_camera_gated_lidar_1c_dbm
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_clear_day.txt -s test_clear_day_camera_gated_lidar_1c_dbm
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_clear_night.txt -s test_clear_night_camera_gated_lidar_1c_dbm
cd ./generic_tf_tools # File Editing
filename="data2example_1c_db.py"
search="'moderate': 0"
replace="'moderate': 1"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" $filename
search="'hard': 0"
replace="'hard': 1"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" $filename
fi
cd ..
# Message
echo "File Edit Completed"
rm nohup.out
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_lightfog_day.txt -s test_lightfog_day_camera_gated_lidar_1c_dbh
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_lightfog_night.txt -s test_lightfog_night_camera_gated_lidar_1c_dbh
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_densefog_day.txt -s test_densefog_day_camera_gated_lidar_1c_dbh
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_densefog_night.txt -s test_densefog_night_camera_gated_lidar_1c_dbh
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_snow_day.txt -s test_snow_day_camera_gated_lidar_1c_dbh
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_snow_night.txt -s test_snow_night_camera_gated_lidar_1c_dbh
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_clear_day.txt -s test_clear_day_camera_gated_lidar_1c_dbh
python create_generic_db2_1c_bg_db.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_clear_night.txt -s test_clear_night_camera_gated_lidar_1c_dbh

cd ./generic_tf_tools # File Editing
filename="data2example_1c_db.py"
search="'hard': 1"
replace="'hard': 0"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" $filename
search="'easy': 0"
replace="'easy': 1"
if [[ $search != "" && $replace != "" ]]; then
sed -i -e "s/$search/$replace/" $filename
fi
cd ..

echo "Completed"
echo "Done"
