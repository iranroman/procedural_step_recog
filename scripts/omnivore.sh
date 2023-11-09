#!/bin/bash

SOURCE_PATH=$1
OUTPUT_PATH=$2               #the same inside $CONFIG_PATH
CONFIG_PATH=$3
squash_files="true"
frame_path="/rgb"            #emtpy if you DO NOT want to process RAW frames in $SOURCE_PATH  
frame_aug_path="/rgb_aug"    #emtpy if you DO NOT want to process AUGMENTED frames in $SOURCE_PATH
squash_root="/omnivore"      #root folder of the squash files

function exec()
{
echo "Sending omnivore batch"

sbatch <<EOSBATCH
#!/bin/bash
#SBATCH -c 1
#SBATCH --mem 64GB
#SBATCH --time 24:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name omnivore-action
#SBATCH --output logs/%J_omnivore-action.out
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=$USER@nyu.edu

if [[ ! -x sing  ]]; then
chmod u+x sing
fi

./sing << EOF

python tools/run_act_recog.py --cfg $CONFIG_PATH

EOF
EOSBATCH

## sed -e 's/$NV --overlay "$OVERLAY:ro" $@ "$SIF"/$NV $@ --overlay "$OVERLAY:ro" "$SIF"/' -i sing
}

function exec_with_squash()
{
echo "|- Action recognition"

for video in "${SEARCH_FOR[@]}"; do

NAME="${video%.*}"
ADD_OVER=""

if [[ ! -z $frame_path ]]; then
  ADD_OVER="$ADD_OVER --overlay $SOURCE_PATH/$frame_path/$video:ro"
fi
if [[ ! -z $frame_aug_path ]]; then
  ADD_OVER="$ADD_OVER --overlay $SOURCE_PATH/$frame_aug_path/$video:ro"
fi

echo "|-- Create SquashFS for $NAME"

TMP=$OUTPUT_PATH/tmp_$NAME

#echo $ADD_OVER
break

sbatch <<EOSBATCH
#!/bin/bash
#SBATCH -c 12
#SBATCH --mem 64GB
#SBATCH --time 24:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name action-$NAME
#SBATCH --output logs/%J_$NAME.out
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=$USER@nyu.edu

if [[ ! -x sing  ]]; then
chmod u+x sing
fi

./sing $ADD_OVER << EOF

python tools/run_act_recog.py --cfg $CONFIG_PATH

EOF

echo "Create SquashFS for $NAME"

TMP=$OUTPUT_PATH/tmp_$NAME

mkdir -p $TMP/$squash_root
mv $OUTPUT_PATH/$NAME* $TMP/$squash_root

find $TMP/$squash_root -type d -exec chmod 755 {} \;
find $TMP/$squash_root -type f -exec chmod 644 {} \;

mksquashfs $TMP$squash_root $OUTPUT_PATH/$NAME.sqf -keep-as-directory -noappend && rm -rv $TMP

EOSBATCH

done
}

if $squash_files == "true"; then

SEARCH_FOR=(
#  "M1-13.sqf"     #OK
#  "M1-14.sqf"     #OK 
#  "M1-15.sqf"     #OK
#  "M1-16.sqf"     #OK
#  "M1-17.sqf"     #OK 
#  "M1-18.sqf"     #OK
#  "M1-19.sqf"     #OK
#  "M1-20.sqf"     #OK
#  "M1-21.sqf"     #OK
#  "M1-22.sqf"     #OK
#  "M1-23.sqf"     #OK
#  "M1-24.sqf"     #OK 
#  "M1-25.sqf"     #OK
#  "M1-26.sqf"     #OK
#  "M1-27.sqf"     #OK
#  "M1-28.sqf"     #OK
#  "M1-29.sqf"     #OK
#  "M1-30.sqf"     #OK
#  "M1-31.sqf"     #OK
#  "M1-32.sqf"     #OK
#  "M1-33.sqf"     ##TEST
#  "M1-34.sqf"     #OK
#  "M1-35.sqf"     #OK 
#  "M1-36.sqf"     #OK
#  "M1-37.sqf"     #OK
#  "M1-39.sqf"     #OK 
#  "M1-40.sqf"     #OK 
#  "M1-41.sqf"     #OK
#  "M1-42.sqf"     #OK 
#  "M1-43.sqf"     ##TEST
#  "M1-44.sqf"     #OK
#  "M1-45.sqf"     #OK 
#  "M1-46.sqf"     #OK
#  "M1-47.sqf"     #OK
#  "M1-57.sqf"     #OK
#  "M1-58.sqf"     #OK 
#  "M1-59.sqf"     #OK
#  "M1-60.sqf"     #OK
#  "M1-61.sqf"     #OK
#  "M1-62.sqf"     #OK
#  "M1-63.sqf"     #OK
#  "M1-64.sqf"     #OK
#  "M1-65.sqf"     ##TEST
#  "M1-66.sqf"     #OK
#  "M1-67.sqf"     #OK
#  "M1-68.sqf"     #OK
#  "M1-69.sqf"     ##TEST
#  "M1-70.sqf"     #OK
#  "M1-71.sqf"     #OK
#  "M1-72.sqf"     ##TEST
#  "M1-73.sqf"     #OK
#  "M1-74.sqf"     #OK
#  "M1-75.sqf"     #OK
#  "M1-76.sqf"     #OK
)

##  sed -e 's/$NV $@ --overlay "$OVERLAY:ro" "$SIF"/$NV --overlay "$OVERLAY:ro" $@ "$SIF"/' -i sing

  exec_with_squash

##  sed -e 's/$NV --overlay "$OVERLAY:ro" $@ "$SIF"/$NV $@ --overlay "$OVERLAY:ro" "$SIF"/' -i sing
else
  exec  
fi
##