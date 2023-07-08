import os
shName = "rec.sh"
f = open(shName, "a+")
#for num in range(20, 164):
for num in range(201, 212):
#for num in range(209, 212):
    data_num = num
    path = "CUDA_VISIBLE_DEVICES=1 python PCsampling_demo.py --datanum={}".format(data_num)
    f.write(path)
    f.write('\n')
f.close()
