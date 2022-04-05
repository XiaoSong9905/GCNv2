#
# Convert ONNX.onnx model to TVM.so model
# Reference: https://tvm.apache.org/docs/tutorial/tvmc_python.html
#

# Step 0: set TVM number of thread
# this help speed up tunning and compiling time
# this step should be done before importning the tvm package
import os
import sys
# Or run export TVM_NUM_THREADS=12
os.environ["TVM_NUM_THREADS"] = str( os.cpu_count() )

from tvm.driver import tvmc

def main():
    # Get input, ouput filename
    if ( len(sys.argv) != 3 or len(sys.argv) != 4 ):
        print("Usage: python3 convert_onnx_to_tvm_by_tvmc.py ONNX_MODEL_FILENAME TVM_MODEL_FILENAME (TVM_TUNNING_LOG)")
        exit(1)

    ONNX_MODEL_FILENAME = sys.argv[1]
    TVM_MODEL_FILENAME   = sys.argv[2]

    print(f"...Convert ONNX model [{ONNX_MODEL_FILENAME}] to TVM model [{TVM_MODEL_FILENAME}]")

    # Step 1: Load ONNX model and convert to relay (TVM internal model representation)
    tvm_relay_model = tvmc.load( ONNX_MODEL_FILENAME )

    # Step 1.5: Tune ONNX model
    # TODO: change -mcpu=??? to your CPU architecture by running `cat /sys/devices/cpu/caps/pmu_name`
    # TODO: use -mcpu=skylake-avx512 if your processor support AVX512 ( check by running `cat /proc/cpuinfo` )
    if ( len(sys.argv) == 4 ):
        tunning_records_log = sys.argv
    else:
        tunning_records_log = ONNX_MODEL_FILENAME + '.tvm-tunning-log.json'
        tvmc.tune( tvm_relay_model, target="llvm -mcpu=skylake", tuning_records=tunning_records_log, enable_autoscheduler = True )

    # Step 2: Compile
    tvm_compiled_model = tvmc.compile( tvm_relay_model, target="llvm -mcpu=skylake", \
        tuning_records=tunning_records_log, package_path=TVM_MODEL_FILENAME )

    # Step 3: Run tunned model with tvm runtime to ensure functionality
    result = tvmc.run( tvm_compiled_model, device="cpu" )

    # Load saved package file to verity model work
    tvm_compiled_model_load = tvmc.TVMCPackage( package_path=TVM_MODEL_FILENAME )
    result = tvmc.run( tvm_compiled_model_load, device="cpu" )


if __name__ == '__main__':
    main()