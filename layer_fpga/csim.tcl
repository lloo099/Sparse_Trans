############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project fpga_attention.prj -upgrade
set_top encoder_kernel
add_files layer.cpp
add_files layer.hpp

#add_files bert_fpga.cpp
#add_files bert_fpga.hpp
add_files /home/enai/Downloads/trans-fat/src/baseline/layer/config.hpp


add_files -tb /home/enai/Downloads/trans-fat/src/baseline/layer/parameters/embedding_in_hls.txt
add_files -tb /home/enai/Downloads/trans-fat/src/baseline/layer/attention_test.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1" -flow_target vivado
set_part {xcu50-fsvh2104-2-e}
create_clock -period 4 -name default
set_directive_top -name encoder_kernel "encoder_kernel"
csim_design
#csynth_design
#cosim_design
#export_design -format ip_catalog
exit
