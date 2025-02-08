echo "Done"
nohup python create_generic_db2_db_1c.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_lightfog_day.txt -s test_lightfog_day_camera_1c_dbe &
nohup python create_generic_db2_db_1c.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_lightfog_night.txt -s test_lightfog_night_camera_1c_dbe &

cd ./generic_tf_tools # File Editing
filename="data2example_1c.py"
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

rm nohup.out
nohup python create_generic_db2_db_1c.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_lightfog_day.txt -s test_lightfog_day_camera_1c_dbm &
nohup python create_generic_db2_db_1c.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_lightfog_night.txt -s test_lightfog_night_camera_1c_dbm &

cd ./generic_tf_tools # File Editing
filename="data2example_1c.py"
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
nohup python create_generic_db2_db_1c.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_lightfog_day.txt -s test_lightfog_day_camera_1c_dbh &
nohup python create_generic_db2_db_1c.py -r /data/datasets/saket/SeeingThroughFogData -d /data/datasets/saket -id SeeingThroughFogData -f /home/saket/Dense/splits/new/test_lightfog_night.txt -s test_lightfog_night_camera_1c_dbh &

cd ./generic_tf_tools # File Editing
filename="data2example_1c.py"
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