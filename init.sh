rm -r patches
rm -r rediction_result
rm -r segmentation_result
rm -r subtraction
rm -r tb
rm -r trained_model

mkdir patches
mkdir rediction_result
mkdir segmentation_result
mkdir subtraction
mkdir tb
mkdir trained_model

python data.py
python make_patch.py
python make_tfrecords_drone.py
python train.py
python recall-precision.py