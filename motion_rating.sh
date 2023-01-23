studylist="studyList"
for study in $(cat $studylist)
do
echo python3 motion_rating_dcm.py $study
python3 motion_rating_dcm.py $study
done
