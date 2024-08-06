#/bin/bash

echo "are you sure?"
read arg
if [ $arg == "yes" ]
then
    rm -rf neb.log nebclimb.log nohup.out
fi
