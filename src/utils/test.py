import os
import subprocess
import sys
import random

def gen_str(prefix, nx, ny, nz):
  dims = 1;
  
  if(ny != 0):
    dims = 2

  if(nz != 0):
    dims = 3 

  dims_str = ""
  file_name = ""
  if(dims == 1):
    dims_str = "-1 " + str(nx)
    file_name = prefix +"_"  + str(nx)

  if(dims == 2):
    dims_str = "-2 " + str(nx) + " " + str(ny)
    file_name = prefix + "_" + str(nx) + "_" + str(ny)

  if(dims == 3):
    dims_str = "-3 " + str(nx) + " " + str(ny) + " " + str(nz)
    file_name = prefix + "_" + str(nx) + "_" + str(ny) + "_" + str(nz)
  return file_name, dims_str
      

def gen_data(generator, scalar_type, nx, ny, nz):

  file_name, dim = gen_str("data", nx, ny, nz)      
  file_name += "_" + scalar_type
  command = generator + " -o " + file_name + " " + dim + " -t " + scalar_type
  print "Executing : " + command
  subprocess.check_call(command, shell=True)
  return file_name 

def compress(compressor, in_file, scalar_type, nx, ny, nz, rate):

  file_name, dim = gen_str("compressed", nx, ny, nz)      
  file_name += "_" + scalar_type + "_r" + str(rate)
  exe = compressor.rsplit("/", 1)[1]
  file_name += "_" + exe 

  command = compressor + " -i " + in_file+ " " + dim 
  command += " -t " + scalar_type + " -r " + str(rate)
  command += " -z " + file_name
  print "Executing : " + command
  subprocess.check_call(command, shell=True)
  return file_name

def fuzz(types, nx, ny, nz, rate):
  for t in types :
    in_data = gen_data(gen, t, nx, ny, nz)
    cmp_cuda = compress(cu_zfp, in_data, t, nx, ny, nz, rate)
    cmp_cpu = compress(zfp, in_data, t, nx, ny, nz, rate)
    compare = "cmp " + cmp_cuda + " " + cmp_cpu
    print "Executing : " + compare
    proc = subprocess.Popen(compare, shell=True)
    proc.wait()
    print "return code ", proc.returncode
    if(proc.returncode != 0):
      print "  **** compression test failed **** "
      exit(1)

def fuzz_1d(num_tests, types):
  for i in range(0, num_tests):
    nx = random.randrange(1, 101, 1) * 4
    ny = 0
    nz = 0 
    #rate = random.randrange(1, 32, 1)
    rate = 19 
    fuzz(types, nx, ny, nz, rate)

if(len(sys.argv) != 3):
  print "Incorrect number of arguments" 
  print "Usage: script path_to_cu_zfp path_to_zfp" 

cu_zfp = sys.argv[1] + "/cuda_zfp"
gen = sys.argv[1] + "/data_gen"
zfp = sys.argv[2] + "/zfp"

#types = ["f32", "f64", "i32", "i64"]
types = ["i32", "i64"]

fuzz_1d(1,types)

