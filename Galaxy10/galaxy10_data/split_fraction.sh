for dir in 0 1 2 3 4 5 6 7 8 9
do
   mkdir -p image_data_train_20/ans_$dir
   counter=1
   for file in image_data_train/ans_$dir/*
   do
      cp $file image_data_train_20/ans_$dir/
      ((counter++))
      if  (($counter%320 == 1))
      then
	      break
      fi
   done
done