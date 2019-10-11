import os
import sys

time = str(sys.argv[1])
n = int(sys.argv[2])

file_prefix = "dumpi-" + time

for i in range(n):
  file_name = file_prefix + "-" + str(i).zfill(4) + ".bin"
  new_name = "n" + str(n) + "-" + str(i).zfill(4) + ".bin"
  print('Renaming', file_name, 'to', new_name)
  os.rename(file_name, new_name)

file_name = file_prefix + ".meta"
new_name = "n" + str(n) + ".meta"
print('Renaming', file_name, 'to', new_name)
os.rename(file_name, new_name)
